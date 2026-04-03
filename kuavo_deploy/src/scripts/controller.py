#!/usr/bin/env python3
"""
KuavoRobot control command transmitter
Used to send control instructions to the running script.py process

Usage example:
  python controller.py pause    #Pause the robot arm movement
  python controller.py resume   #Restoring robotic arm motion
  python controller.py stop     #Stop the robot arm movement
  python controller.py status   #View process status
"""

import os
import sys
import signal
import psutil
import argparse
from pathlib import Path

def find_example_process():
    """
    Find the running script.py process
    
    Returns:
        psutil.Process: The process object found, returns None if not found
    """
    target_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            #Check process name or command line arguments
            if (proc.info['name'] == 'python' or proc.info['name'] == 'python3') and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                
                #Exactly match the kuavo_deploy/src/scripts/script.py path
                if 'kuavo_deploy/src/scripts/script.py' in cmdline:
                    target_processes.append((proc, 'exact'))
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    if len(target_processes) != 1:
        print(f"❌ Found {len(target_processes)} matching processes, please use the --pid parameter to specify the process ID")
        sys.exit(1)
    else:
        return target_processes[0][0]

def send_signal_to_process(proc, signal_type):
    """
    Send a signal to the specified process
    
    Args:
        proc: psutil.Processobject
        signal_type: signal type ('pause', 'resume', 'stop')
    """
    try:
        if signal_type == 'pause':
            proc.send_signal(signal.SIGUSR1)
            print(f"✅ A pause signal has been sent to process {proc.pid}")
        elif signal_type == 'resume':
            proc.send_signal(signal.SIGUSR1)
            print(f"✅ Resume signal sent to process {proc.pid}")
        elif signal_type == 'stop':
            proc.send_signal(signal.SIGUSR2)
            print(f"✅ Stop signal sent to process {proc.pid}")
        else:
            print(f"❌ Unknown signal type: {signal_type}")
            return False
        return True
    except psutil.NoSuchProcess:
        print(f"❌ Process {proc.pid} does not exist")
        return False
    except psutil.AccessDenied:
        print(f"❌ No permission to send signal to process {proc.pid}")
        return False
    except Exception as e:
        print(f"❌ An error occurred while sending the signal: {e}")
        return False

def show_process_status(proc):
    """
    Display process status information
    
    Args:
        proc: psutil.Processobject
    """
    try:
        print(f"📊 Process information:")
        print(f"  PID: {proc.pid}")
        print(f"Status: {proc.status()}")
        print(f"Creation time: {proc.create_time()}")
        print(f"CPU usage: {proc.cpu_percent()}%")
        print(f"Memory usage: {proc.memory_info().rss/1024/1024:.1f} MB")
        print(f"  Command line: {' '.join(proc.cmdline())}")
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        print(f"❌ Unable to obtain process information: {e}")

def main():
    """main function"""
    parser = argparse.ArgumentParser(
        description="Kuavo robot control command transmitter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage example:
  python controller.py pause    #Pause the robot arm movement
  python controller.py resume   #Restoring robotic arm motion
  python controller.py stop     #Stop the robot arm movement
  python controller.py status   #View process status

Control instruction description:
  pause   - Pause the robot arm movement (send SIGUSR1 signal)
  resume  - Resume robot arm movement (send SIGUSR1 signal)
  stop    - Stop the robot arm movement (send SIGUSR2 signal)
  status  - Display the status of the currently running script.py process
        """
    )
    
    parser.add_argument(
        "command",
        type=str,
        choices=["pause", "resume", "stop", "status"],
        help="control instructions"
    )
    
    parser.add_argument(
        "--pid",
        type=int,
        help="Specify the process PID (if not specified, the script.py process will be automatically found)"
    )
    
    args = parser.parse_args()
    
    #Find target process
    target_proc = None
    
    if args.pid:
        #Use specified PID
        try:
            target_proc = psutil.Process(args.pid)
            #Verify that the process is running script.py
            cmdline = ' '.join(target_proc.cmdline())
            if 'script.py' not in cmdline:
                print(f"❌ The process {args.pid} is not the script.py process")
                print(f"Command line: {cmdline}")
                sys.exit(1)
        except psutil.NoSuchProcess:
            print(f"❌ Process {args.pid} does not exist")
            sys.exit(1)
        except psutil.AccessDenied:
            print(f"❌ No permission to access process {args.pid}")
            sys.exit(1)
    else:
        #Automatically find script.py process
        print("🔍 Looking for running script.py processes...")
        target_proc = find_example_process()
        
        if not target_proc:
            print("❌ The running script.py process was not found")
            print("💡 Please make sure script.py is running, or use the --pid parameter to specify the process ID")
            print("💡 Expected process path: kuavo_deploy/src/scripts/script.py")
            sys.exit(1)
        
        #Display found process information
        cmdline = ' '.join(target_proc.cmdline())
        if 'kuavo_deploy/src/scripts/script.py' in cmdline:
            print(f"✅ Find the exact matching process: {target_proc.pid}")
        else:
            print(f"⚠️ Found partial matching process: {target_proc.pid}")
            print(f"Command line: {cmdline}")
    
    #execute command
    if args.command == "status":
        show_process_status(target_proc)
    else:
        print(f"🎯 Target process: {target_proc.pid}")
        success = send_signal_to_process(target_proc, args.command)
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()
