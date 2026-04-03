import sys
import time
import termios
import tty
import select

class KeyListener:
    def __init__(self):
        self.exit_program = False
        self.key_callbacks = {}
        self.crtk_c_callback = None
        self.old_settings = termios.tcgetattr(sys.stdin)

    def register_ctrlC_callback(self, callback):
        self.crtk_c_callback = callback    

    def register_callbacks(self, keys, callback):
        """Register buttons and corresponding callback functions"""
        for key in keys:
            self.key_callbacks[key] = callback

    def register_callback(self, key, callback):
        """Register buttons and corresponding callback functions"""
        self.key_callbacks[key] = callback

    def unregister_callback(self, key):
        """Callback function for logout key"""
        if key in self.key_callbacks:
            del self.key_callbacks[key]

    def getKey(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        return key
    
    def on_press(self, key):
        try:
            if key in self.key_callbacks and callable(self.key_callbacks[key]):
                self.key_callbacks[key](key)
        except AttributeError:
            #Some special keys (such as function keys) may not have character attributes
            pass
        except Exception as e:
            print(f"Error processing key: {e}")
        # print("pressed key: '", key, "'",end='\r')
    def stop(self):
        self.exit_program = True
    def loop_control(self):
        try:
            while not self.exit_program:
                key = self.getKey()
                if key:
                    self.on_press(key)
                if (key == '\x03'):  # Ctrl-C
                    if self.crtk_c_callback and callable(self.crtk_c_callback):
                        self.crtk_c_callback()
                    break 
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)    
if __name__ == "__main__":
    kl = KeyListener()

    #Register key callback function
    kl.register_callback('w', lambda key: print("pressed key: '", key, "'",end='\r'))
    kl.register_callback('s', lambda key: print("pressed key: '", key, "'",end='\r'))
    kl.register_callback('a', lambda key: print("pressed key: '", key, "'",end='\r'))
    kl.register_callback('d', lambda key: print("pressed key: '", key, "'",end='\r'))
    kl.register_callback('+', lambda key: print("pressed key: '", key, "'",end='\r'))
    kl.register_callback('-', lambda key: print("pressed key: '", key, "'",end='\r'))
    kl.register_callback('=', lambda key: print("pressed key: '", key, "'",end='\r'))
    
    try:
        kl.loop_control()
    except KeyboardInterrupt:
        kl.stop()
    