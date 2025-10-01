# pip install nidaqmx matplotlib
import time
from collections import deque
import sys

import nidaqmx
from nidaqmx.errors import DaqError
import matplotlib.pyplot as plt

# ========= USER: set your saved DAQmx task name =========
TASK_NAME = "My_Current_Task"   # must match the task in NI-MAX exactly
# ========================================================

# ---- user-tunable display defaults ----
WINDOW_SECONDS = 60        # plot ~this much history
UPDATE_BLOCKS_PER_SEC = 1  # set to 5 for snappier ~200 ms updates if your rate allows

def _ensure_list(x):
    return x if isinstance(x, list) else [x]

def _prompt_indices(prompt, valid, default=None):
    """
    Ask for comma-separated indices from 'valid' (iterable of ints).
    Empty input -> default (may be a list or None).
    """
    while True:
        s = input(prompt).strip()
        if s == "" and default is not None:
            return list(default)
        if not s:
            print("Please enter something (e.g., 0,1,2) or press Enter for default.")
            continue
        try:
            idxs = [int(tok) for tok in s.replace(" ", "").split(",") if tok != ""]
            if not idxs:
                raise ValueError
            if any(i not in valid for i in idxs):
                raise ValueError
            # de-dup but preserve order
            seen = set()
            out = []
            for i in idxs:
                if i not in seen:
                    seen.add(i)
                    out.append(i)
            return out
        except ValueError:
            print(f"Invalid input. Valid indices: {list(valid)}")

def _choose_channels(task):
    chans = task.ai_channels
    n = len(chans)
    names = [ch.name for ch in chans]          # DAQmx virtual names (often nicer)
    phys  = [ch.physical_name for ch in chans] # e.g., 'cDAQ1Mod1/ai0'

    print("\nAvailable AI channels in task:")
    for i, (nm, ph) in enumerate(zip(names, phys)):
        print(f"  [{i}] {nm}  ({ph})")

    all_idxs = list(range(n))
    plot_idxs = _prompt_indices(
        f"\nEnter indices to plot (comma-separated) [default: all {all_idxs}]: ",
        valid=all_idxs, default=all_idxs
    )

    # Ask which of the selected should go on secondary y-axis
    sec_idxs = _prompt_indices(
        f"Enter indices (from your selection) to plot on SECONDARY Y-axis "
        f"(comma-separated) [default: none]: ",
        valid=plot_idxs, default=[]
    )

    pri_idxs = [i for i in plot_idxs if i not in sec_idxs]
    return names, phys, pri_idxs, sec_idxs

def main():
    try:
        with nidaqmx.Task(TASK_NAME) as task:
            # Discover rate (fallback if not configured)
            try:
                rate = float(task.timing.samp_clk_rate)
                if rate <= 0:
                    rate = 5.0
            except Exception:
                rate = 5.0

            # Determine block size for each read/plot cycle
            read_n = max(1, int(round(rate / max(1, UPDATE_BLOCKS_PER_SEC))))

            names, phys, pri_idxs, sec_idxs = _choose_channels(task)
            nchan = len(names)
            if not pri_idxs and not sec_idxs:
                print("No channels selected. Exiting.")
                return

            print("\n--- Acquisition Plan ---")
            print(f"Sample rate ~ {rate:.3f} Hz  |  read block size = {read_n} sample(s)/chan/update")
            print("Primary Y-axis channels :", [names[i] for i in pri_idxs] if pri_idxs else "None")
            print("Secondary Y-axis channels:", [names[i] for i in sec_idxs] if sec_idxs else "None")
            print("NOTE: TDMS logging is performed by DAQmx using the file/path set in the task.")
            print("Press Ctrl+C to stop.\n")

            # ---- plotting setup ----
            plt.ion()
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx() if sec_idxs else None

            # Prepare buffers sized for ~WINDOW_SECONDS
            window_len = max(read_n, int(rate * WINDOW_SECONDS))
            tbuf = deque(maxlen=window_len)
            ybufs = [deque(maxlen=window_len) for _ in range(nchan)]

            # Make line objects
            lines = [None] * nchan
            # Primary axis
            for i in pri_idxs:
                (line,) = ax1.plot([], [], label=names[i])
                lines[i] = line
            # Secondary axis
            if ax2:
                for i in sec_idxs:
                    (line,) = ax2.plot([], [], label=names[i])
                    lines[i] = line

            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Primary axis (engineering units)")
            if ax2:
                ax2.set_ylabel("Secondary axis (engineering units)")

            # Combined legend
            if ax2:
                labs = [names[i] for i in pri_idxs] + [names[i] for i in sec_idxs]
                handles = [lines[i] for i in pri_idxs] + [lines[i] for i in sec_idxs]
                ax1.legend(handles, labs, loc="upper right")
            else:
                ax1.legend(loc="upper right")

            ax1.grid(True)

            # Start task (this is when DAQmx begins TDMS logging if enabled in the task)
            task.start()
            start = time.time()

            while True:
                # Read a block. For multi-chan, DAQmx returns list-of-lists [chan][samples]
                data = task.read(number_of_samples_per_channel=read_n, timeout=5.0)

                # Normalize shapes
                if nchan == 1:
                    per_chan = [ _ensure_list(data) ]  # -> [ [samples] ]
                else:
                    per_chan = [ _ensure_list(ch) for ch in data ]

                block_len = len(per_chan[0])
                now = time.time() - start

                # Append into buffers with correct timestamps per sample in block
                for k in range(block_len):
                    tbuf.append(now - (block_len - 1 - k) / rate)
                    for i in range(nchan):
                        ybufs[i].append(per_chan[i][k])

                # Update plot data
                for i in pri_idxs:
                    lines[i].set_xdata(tbuf)
                    lines[i].set_ydata(ybufs[i])
                if ax2:
                    for i in sec_idxs:
                        lines[i].set_xdata(tbuf)
                        lines[i].set_ydata(ybufs[i])

                # Rescale views
                ax1.relim(); ax1.autoscale_view()
                if ax2:
                    ax2.relim(); ax2.autoscale_view()

                plt.pause(0.001)

    except KeyboardInterrupt:
        print("\nStopping…")
    except DaqError as e:
        print(f"\nDAQmx error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()