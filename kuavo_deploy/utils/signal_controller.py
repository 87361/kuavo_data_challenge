from kuavo_deploy.utils.logging_utils import setup_logger
from std_msgs.msg import Bool
from kuavo_deploy.utils.ros_manager import ROSManager
import threading, time


log_robot = setup_logger("robot")

class ControlSignalManager:
    """control signal manager"""
    def __init__(self):
        self.ros_manager = ROSManager()
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Set up signal handler"""
        self.ros_manager.register_subscriber('/kuavo/pause_state', Bool, self._pause_callback)
        self.ros_manager.register_subscriber('/kuavo/stop_state', Bool, self._stop_callback)
    
    def _pause_callback(self, msg):
        """Pause callback"""
        if msg.data:
            self.pause_flag.set()
        else:
            self.pause_flag.clear()

    def _stop_callback(self, msg):
        """Stop callback"""
        if msg.data:
            self.stop_flag.set()
    
    def check_control_signals(self):
        """Check control signals"""
        #Check pause status
        while self.pause_flag.is_set():
            log_robot.info("🔄 Robot arm motion paused")
            time.sleep(0.1)
            if self.stop_flag.is_set():
                log_robot.info("🛑 Robot arm motion stopped")
                return False
        
        #Check if it needs to be stopped
        if self.stop_flag.is_set():
            log_robot.info("🛑 Stop signal detected, exiting arm motion")
            return False
            
        return True  #Continue normally Continue
    
    def close(self):
        """Release Resources Release Resources"""
        self.ros_manager.close()
        self.stop_flag.clear()
        self.pause_flag.clear()
