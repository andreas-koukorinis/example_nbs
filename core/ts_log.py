import logging
import datetime
from textwrap import TextWrapper
import threading
import pytz


_default_log_file = None
_suppress_stdout = False
_mode = 'a'


class Logger(object):
    """
    Wraps a usual python logger to add timeseries capabilities
    """
    def __init__(self, logger, log_thread=False):
        self._logger = logger
        self._log_thread = log_thread
        self._wrapper = TextWrapper(180)
        self.MAX_NAME_LENGTH = 12

    def info(self, timestamp, log):
        if not self._logger.isEnabledFor(logging.INFO):
            return
        if timestamp is not None:
            if len(log) > 180:
                log_lines = self._wrapper.wrap(log)
                for log_line in log_lines:
                    self._logger.info(self._format_message(timestamp, log_line))
            else:
                self._logger.info(self._format_message(timestamp, log))
        else:
            if len(log) > 180:
                log_lines = self._wrapper.wrap(log)
                for log_line in log_lines:
                    self._logger.info(self._format_message(timestamp, log_line))
            else:
                self._logger.info(self._format_message(timestamp, log))

    def set_level(self, log_level):
        self._logger.setLevel(log_level)

    def warn(self, timestamp, log):
        if not self._logger.isEnabledFor(logging.WARN):
            return
        self._logger.warn(self._format_message(timestamp, log))

    def error(self, timestamp, log):
        if not self._logger.isEnabledFor(logging.ERROR):
            return
        self._logger.error(self._format_message(timestamp, log))

    def debug(self, timestamp, log):
        if not self._logger.isEnabledFor(logging.DEBUG):
            return
        self._logger.debug(self._format_message(timestamp, log))

    def _format_name(self, name):
        if not name:
            return ' ' * self.MAX_NAME_LENGTH
        name = name.split('.')[-1][:self.MAX_NAME_LENGTH]
        name += ' ' * (self.MAX_NAME_LENGTH - len(name))
        return name

    def _format_timestamp(self, timestamp):
        if timestamp.tzinfo:
            timestamp = timestamp.astimezone(pytz.UTC)

        s = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
        s = s[:-3]  # just keep down to the millisecond
        if timestamp.tzinfo:
            s += 'Z'
        return s

    def _format_message(self, timestamp, message):
        if self._log_thread:
            message = " | %s | %s | %s" % (threading.currentThread().getName(),
                                           self._format_name(self._logger.name), message)

        if timestamp is not None:
            return '%s|%s|%s %s' % (self._format_timestamp(datetime.datetime.now()),
                                    self._format_timestamp(timestamp),
                                    self._format_name(self._logger.name), message)
        else:
            return '%s|%s %s' % (self._format_timestamp(datetime.datetime.now()),
                                 self._format_name(self._logger.name), message)

    def __deepcopy__(self, memo):
        return Logger(logging.getLogger(self._logger.name), log_thread=self._log_thread)


def set_default_log_file(fname):
    global _default_log_file
    _default_log_file = fname


def set_suppress_stdout_flag(flag, default_log_file=None, mode='a'):
    global _suppress_stdout, _default_log_file
    _suppress_stdout = flag
    _default_log_file = default_log_file
    _mode = mode


def get_logger(name, fname=None, log_thread=False):
    logger = logging.getLogger(name)

    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[0])

    logger.setLevel(logging.INFO)
    if not _suppress_stdout:
        ch = logging.StreamHandler()
        logger.addHandler(ch)

    logger.propagate = False

    fname = fname or _default_log_file

    if fname is not None:
        fh = logging.FileHandler(fname, mode=_mode)
        logger.addHandler(fh)

    result = Logger(logger, log_thread)

    # result.set_level(logging.CRITICAL)
    return result