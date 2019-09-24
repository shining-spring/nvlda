import threading
import sys
import os
if sys.version_info.major == 2:
    import ConfigParser
elif sys.version_info.major == 3:
    import configparser as ConfigParser

def isint(s):
    try:
        float(s)
        return True
    except:
        return False


def isiterable(obj):
    try:
        iter(obj)
        return True
    except:
        return False


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

class Configs(object):
    def __init__(self):
        return

def read_config_section(config, section):
    setting = Configs()
    for key, value in config.items(section):
        setting.__setattr__(key, eval(value))

    if hasattr(setting, "othersettings"):
        for att in setting.othersettings:
            att_setting = read_config_section(config, att)
            for key, item in att_setting.__dict__.iteritems():
                setting.__setattr__(key, item)
    return setting

def read_config(config_file, section="DEFAULT"):
    config = ConfigParser.RawConfigParser(allow_no_value=True)
    config.optionxform = str
    config.read(config_file)
    return read_config_section(config, section)
