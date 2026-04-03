import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from termcolor import colored

class ColoredFormatter(logging.Formatter):
    """Customized color log formatter Customized color log formatter"""
    
    #Default style configuration
    DEFAULT_STYLE_CONFIG = {
        'env': {
            'tag': '🤖 ENV',
            'DEBUG': {'color': 'grey', 'attrs': ['dark']},
            'INFO': {'color': 'green', 'attrs': []},
            'WARNING': {'color': 'yellow', 'attrs': ['bold']},
            'ERROR': {'color': 'red', 'attrs': ['bold']},
            'CRITICAL': {'color': 'white', 'attrs': ['bold'], 'on_color': 'on_red'}
        },
        'model': {
            'tag': '🧠 MODEL',
            'DEBUG': {'color': 'grey', 'attrs': ['dark']},
            'INFO': {'color': 'blue', 'attrs': ['bold']},
            'WARNING': {'color': 'magenta', 'attrs': ['bold']},
            'ERROR': {'color': 'red', 'attrs': ['bold']},
            'CRITICAL': {'color': 'white', 'attrs': ['bold'], 'on_color': 'on_red'}
        },
        'robot': {
            'tag': '🦾 ROBOT',
            'DEBUG': {'color': 'cyan', 'attrs': ['dark']},
            'INFO': {'color': 'green', 'attrs': []},
            'WARNING': {'color': 'yellow', 'attrs': ['bold']},
            'ERROR': {'color': 'red', 'attrs': ['bold']},
            'CRITICAL': {'color': 'red', 'attrs': ['bold', 'underline', 'blink']}
        }
    }
    
    def __init__(self, fmt: str, style_config: Dict = None):
        super().__init__(fmt)
        self.style_config = style_config or self.DEFAULT_STYLE_CONFIG
        self.is_console = False  #File output by default File output by default

    def format(self, record):
        #Cache original message, as it will be modified later on record.msg Cache original message, as it will be modified later on
        original_msg = record.msg
        
        #Fetch corresponding module configuration Fetch corresponding module configuration
        source_config = self.style_config.get(record.name, {})
        source_tag = source_config.get('tag', f'📝 {record.name.upper()}')
        style = source_config.get(record.levelname, {'color': 'white', 'attrs': []})
        
        #Build location information (file name: line number) - refer to the method Construct location info of ks_download.py
        location_info = ""
        if hasattr(record, 'pathname') and hasattr(record, 'lineno'):
            fnameline = f"{record.pathname}:{record.lineno}"
            #Truncate the last 20 characters and right-align, slightly longer than ks_download.py to display more information
            # location_info = f" {fnameline[-20:]:>20}"
            location_info = f" {fnameline}"
        
        #Construct messages Construct messages
        if hasattr(self, 'is_console') and self.is_console:
            #Color on console output Add color Color on console output
            colored_message = colored(
                f"{record.levelname}: {original_msg}",
                color=style['color'],
                on_color=style.get('on_color'),
                attrs=style['attrs']
            )
            record.msg = f"{source_tag} | {colored_message} |{location_info} "
        else:
            #No color for file output
            record.msg = f"{source_tag} | {record.levelname}: {original_msg} | {location_info} "
        #Formatted message Formatted message
        formatted_message = super().format(record)
        
        #Restore original message Restore original message
        record.msg = original_msg
        
        return formatted_message

class LoggerManager:
    def __init__(self, 
                 log_dir: Optional[str] = None, 
                 log_level: str = "INFO",
                 custom_loggers: Optional[Dict] = None,
                 save_to_file: bool = False):
        """
        Initialize the log manager
        
        Args:
            log_dir: Log storage directory
            log_level: Log level
            custom_loggers: Custom logger configuration
                For example: {
                    'other': {
                        'tag': '👁️ OHTER',
                        'DEBUG': {'color': 'grey'},
                        'INFO': {'color': 'blue'},
                        ...
                    }
                }
            save_to_file: Whether to save logs to file,Default is False
        """
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = self._setup_log_dir(log_dir) if save_to_file else None
        self.loggers = {}
        
        #Merge custom logger configuration
        self.style_config = ColoredFormatter.DEFAULT_STYLE_CONFIG.copy()
        if custom_loggers:
            self.style_config.update(custom_loggers)

        #If you need to save to a file, create a unified file processor
        self.file_handler = None
        if save_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.file_handler = logging.FileHandler(
                self.log_dir / f"kuavomimic_{timestamp}.log",
                encoding='utf-8'
            )
            #File processor uses colorless formatter
            file_formatter = ColoredFormatter(
                '%(asctime)s - %(message)s',
                style_config=self.style_config
            )
            file_formatter.is_console = False
            self.file_handler.setFormatter(file_formatter)

    def _setup_log_dir(self, log_dir: Optional[str]) -> Path:
        if log_dir is None:
            log_dir = Path.cwd() / 'logs'
        else:
            log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create logger"""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(self.log_level)
            logger.handlers.clear()
            
            #Console Processor (Color)
            console_handler = logging.StreamHandler()
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(message)s',
                style_config=self.style_config
            )
            console_formatter.is_console = True  #Mark as console output
            console_handler.setFormatter(console_formatter)
            
            logger.addHandler(console_handler)
            if self.file_handler:
                logger.addHandler(self.file_handler)
            
            self.loggers[name] = logger
            
        return self.loggers[name]

#Global log manager instance
_log_manager = None

def get_log_manager(log_dir: Optional[str] = None, 
                   log_level: str = "INFO",
                   custom_loggers: Optional[Dict] = None,
                   save_to_file: bool = False) -> LoggerManager:
    """Get global log manager instance"""
    global _log_manager
    if _log_manager is None:
        _log_manager = LoggerManager(log_dir, log_level, custom_loggers, save_to_file)
    return _log_manager

def setup_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None, save_to_file: bool = False) -> logging.Logger:
    """
    Sets and returns a named logger
    
    Args:
        name: Logger name
        level: Log level
        log_file: Optional log file path
        save_to_file: Whether to save logs to file,Default is False
        
    Returns:
        Configured logger
    """
    #Get global log manager
    log_manager = get_log_manager(log_dir=None, log_level="INFO", custom_loggers=None, save_to_file=save_to_file)
    
    #Get or create logger
    logger = log_manager.get_logger(name)
    logger.setLevel(level)
    
    #If a specific log file is provided, add additional file handlers
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def highlight_message(logger, message, color="magenta", attrs=None):
    """Highlight messages using custom colors and properties"""
    if attrs is None:
        attrs = ["bold"]
    print(colored(f">>> {message} <<<", color=color, attrs=attrs))
    return logger.info(message)

def test_logging():
    """Test log function"""
    #Create log directory
    log_dir = Path.cwd() / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Log files will be saved in: {log_dir}")
    
    #Optional: define a custom logger
    custom_loggers = {
        'other': {
            'tag': '👋 OTHER',
            'DEBUG': {'color': 'grey', 'attrs': ['dark']},
            'INFO': {'color': 'cyan', 'attrs': ['bold']},
            'WARNING': {'color': 'yellow', 'attrs': ['bold']},
            'ERROR': {'color': 'red', 'attrs': ['bold']},
            'CRITICAL': {'color': 'white', 'attrs': ['bold'], 'on_color': 'on_red'}
        }
    }
    
    #Create a log manager
    log_manager = LoggerManager(log_dir=str(log_dir), log_level="DEBUG", custom_loggers=custom_loggers)
    
    #Get loggers
    env_logger = log_manager.get_logger("env")
    model_logger = log_manager.get_logger("model")
    robot_logger = log_manager.get_logger("robot")
    other_logger = log_manager.get_logger("other")  #Custom logger
    
    #Test log
    env_logger.info("Environment initialization completed")
    model_logger.warning("Model performance degrades")
    robot_logger.info("The robot status is normal")
    other_logger.info("Process camera data")
    env_logger.error("Collision risk detected")
    
    #Testing setup_logger function - not saving to file
    test_logger = setup_logger("test", logging.DEBUG, save_to_file=False)
    test_logger.debug("This is a test log (console output only)")
    test_logger.info("Test information (console output only)")
    
    #Test setup_logger function - save to file
    test_logger_with_file = setup_logger("test_file", logging.DEBUG, save_to_file=True)
    test_logger_with_file.debug("This is a test log (output to file at the same time)")
    test_logger_with_file.info("Test information (output to file at the same time)")
    
    #Test highlighted messages
    highlight_message(test_logger, "This is a highlight message")
    
    ##Print log file path
    # log_files = list(log_dir.glob("*.log"))
    # if log_files:
    #     print(f"Log file created: {[str(log_files) for f in log_files]}")
    # else:
    #     print("Warning: Log file not found!")

if __name__ == "__main__":
    test_logging()
