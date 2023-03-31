from .log import LogStream
from rombus.exceptions import handle_exception

log = LogStream(exception_handler=handle_exception)
