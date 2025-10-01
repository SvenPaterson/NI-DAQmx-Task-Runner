import inspect
import pkgutil
import nidaqmx
for importer, modname, ispkg in pkgutil.walk_packages(nidaqmx.__path__, prefix='nidaqmx.'):
    if modname.endswith('in_stream'):
        module = __import__(modname, fromlist=['InStream'])
        InStream = getattr(module, 'InStream', None)
        if InStream:
            print(modname)
            print(inspect.getsource(InStream.configure_logging))
            break
