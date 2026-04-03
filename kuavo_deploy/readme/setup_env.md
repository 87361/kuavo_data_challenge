#Environment configuration plan

## Install ROS Noetic (ROS 1), not ROS 2!
<!-- https://blog.csdn.net/m0_73745340/article/details/135281023 -->
### Configure installation source
- Open a terminal and type:
```bash
wget http://fishros.com/install -O fishros && . fishros
```
- Then a menu interface will appear; select 5 to configure the system source with one click, test which system source is the most reliable, and choose which one

- After testing, enter 2 to replace the system source and clean up the third-party source

- Then select 1 to add ROS/ROS2 source

### One-click installation of Yuxiang ROS
- Run the same command as above again to open the menu of Yuxiang ROS:
```bash
wget http://fishros.com/install -O fishros && . fishros
```
- This time choose 1 for one-click installation, and then 2 for installation without changing the source.
- After a while, you will be asked which ROS version you want to install. Select **ROS1 Noetic** here, don’t make a mistake! Then select the desktop version
The subsequent automatic installation process will take a long time. Pay attention to see if there are any errors, and open the system resource monitor to see if there is continued network, CPU, or other resource usage. If there is no activity in the terminal and system resources for a long time, you may need to Ctrl+C to cancel and start again.
- If no error is reported after the operation is completed, ROS Noetic is successfully installed.

### Test ROS installation
After the installation is complete, you can use Turtlesim that comes with ROS to test whether ROS can run correctly.
- Start the ros core before opening Turtlesim:
```bash
roscore
```
- Open two more terminals and run:
```bash
rosrun turtlesim turtlesim_node
rosrun turtlesim turtle_teleop_key
```
At this time, there is a window with a small turtle in it, and another window can control the turtle through the keyboard. At this time, there is no problem with the installation!

---
- Download the Miniconda installation script and temporarily download it to the project directory:
```bash  
#From: https://web.archive.org/web/20231129185127/https://mediawiki.middlebury.edu/CS/Useful_Tools
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

- The pre-downloaded Miniconda installation should be here, now perform the installation:
```bash  
bash Miniconda3-latest-Linux-x86_64.sh
```
- First there will be a Terms of Use, press Enter to open it, continue to press it to scroll to the bottom, hit yes and press Enter
- The default directory will be displayed below, the default directory is fine
- If there is another Python Path warning later, hit yes and press Enter again.
- In the following ab two scenarios, create a python environment:
## a. Recreate the Conda environment

* **Create a conda environment (Python 3.10 recommended)**
  ```bash
  # Set "kdc" to your conda env name
  conda create -n kdc python=3.10 #Keep on typing 'a' and enter to accept ToS
  conda activate kdc
  ```


* **For full system (data transformation, simulator, deployment on real robot, etc.):**

  ```bash
  pip install -r requirements_total.txt
  ```

* **For imitation learning training only:**

  ```bash
  pip install -r requirements_ilcode.txt
  ```

## b. Use the packaged environment
```bash
# Also need the download link of kdc_v0.tar.gz
# Or conda unpack the packaged environment. Note that the setup_env.sh script and the environment compressed package file kdc_v0.tar.gz are placed in the same directory.
./setup_env.sh
source ./kdc_v0/bin/activate
```
## Continue to install dependencies
```bash
# Install dependencies. This will take a while...
Enter the third_party/lerobot directory
pip install -e ".[aloha, pusht]"

# Uninstall torchcodec 
pip uninstall torchcodec

Enter the root directory of the kuavo_data_challenge project
pip install -e .

#Install the kuavo_humanoid_sdk package for communication, reference link:
# https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk

```
