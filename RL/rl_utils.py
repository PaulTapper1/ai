import os

if not os.getcwd().endswith("data"):
  os.chdir("data")
  print(F"Set current folder to {os.getcwd()}")

class SettingsIterator:
  def __init__(self, settings_options):
    self.settings_options = settings_options
    self.iterator_cursor = [0]*len(self.settings_options)
    self.settings = self.get_settings_from_cursor()

  def get_settings_from_cursor(self):
    ret = []
    for layer in range(len(self.settings_options)):
      ret.append( self.settings_options[layer][self.iterator_cursor[layer]] )
    return ret

  def iterate_inner(self):  # returns True if it is still iterating, and False when its finished
    if self.iterator_cursor[0] == -1:  # returned last iteration previous time this was called, so now time to terminate loop
      return False
    cursor_layer = 0
    while True:
      if cursor_layer == len(self.settings_options):
        self.iterator_cursor[0] = -1  # will cause a loop termination next time
        break
      self.iterator_cursor[cursor_layer] += 1
      if self.iterator_cursor[cursor_layer] < len(self.settings_options[cursor_layer]):
        break
      self.iterator_cursor[cursor_layer] = 0
      cursor_layer += 1
    return True

  def iterate(self):  # returns True if it is still iterating, and False when its finished
    self.settings = self.get_settings_from_cursor()
    return self.iterate_inner()

class MetaState():
  def __init__(self, names = []):
    self.meta_state = {}
    self.names = names
    self.add_state_names(names)
  
  def add_state_names(self, names):
    for name in names:
      if not name in self.meta_state:
        self.meta_state[name] = []
        
  def add_value(self, name, value):
    if not name in self.meta_state:
      self.meta_state[name] = []    
    self.meta_state[name].append(value)
    
  def get_latest_value(self, name):
    if len(self.meta_state[name]) > 0:
      return self.meta_state[name][-1]
    return 0
  
  def get_values(self,name):
    return self.meta_state[name]
  
  def __len__(self):
    return len(self.meta_state[self.names[0]])
    
  def to_dict(self):
    return { "meta_state":self.meta_state, "names":self.names }
    
  @classmethod
  def from_dict(cls,dict_in):
    ret = cls()
    ret.meta_state = dict_in.get("meta_state")
    ret.names = dict_in.get("names")
    return ret
