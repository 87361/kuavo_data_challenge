# Side machine communication configuration solution tutorial (based on the lower machine bridging the side machine and the upper machine)

---

## 1. Use a network cable + USB port to connect to the lower computer and side computer, and check the wired network port information of the lower computer.

- View wired network interface:

```bash
nmcli connection show
```

Get wired network port information similar to the following

```bash
NAME                   UUID                                  TYPE      DEVICE              
enxc8a362b260f5        871ce4e7-3633-47ab-a88b-c0613c6ca67a  ethernet  enxc8a362b260f5 
Wired connection 2     65e0a36e-e998-38cd-ad8c-a73ad59e10c0  ethernet  enx00e04c684355 
```

---

## 2. Create and configure the bridge interface

Assume that the network port of the lower computer is:

- `enx00e04c684355` (connected to side machine)
- `enxc8a362b260f5` (connected to host computer)

### Create a new bridge configuration script `setup_bridge.sh`, be careful not to duplicate the bridge interface name with an existing one

```bash
#!/bin/bash

# Bridge interface name
BRIDGE=br0

#Physical network card name (replace with your interface name)
IFACE1=enx00e04c684355
IFACE2=enxc8a362b260f5

# Check if the bridge interface exists
if ip link show "$BRIDGE" &>/dev/null; then
    echo "Error: Bridge interface $BRIDGE already exists. Please delete it first or use another name."
    exit 1
fi

# Check if the physical network card exists
if ! ip link show "$IFACE1" &>/dev/null; then
    echo "Error: The physical network card $IFACE1 does not exist, exit the script."
    exit 1
fi

if ! ip link show "$IFACE2" &>/dev/null; then
    echo "Error: The physical network card $IFACE2 does not exist, exit the script."
    exit 1
fi

# Bridge IP address
BRIDGE_IP=192.168.26.1/24

echo "=== Disable interface ==="
sudo ip link set dev $IFACE1 down
sudo ip link set dev $IFACE2 down

echo "=== Clear interface IP address ==="
sudo ip addr flush dev $IFACE1
sudo ip addr flush dev $IFACE2

echo "=== Create bridge interface ==="
sudo ip link add name $BRIDGE type bridge

echo "=== Add physical interface to bridge ==="
sudo ip link set dev $IFACE1 master $BRIDGE
sudo ip link set dev $IFACE2 master $BRIDGE

echo "=== Start physical interface and bridge interface ==="
sudo ip link set dev $IFACE1 up
sudo ip link set dev $IFACE2 up
sudo ip link set dev $BRIDGE up

echo "=== Assign IP address to bridge interface ==="
sudo ip addr add $BRIDGE_IP dev $BRIDGE

echo "=== Display interface status ==="
ip addr show $BRIDGE
ip addr show $IFACE1
ip addr show $IFACE2

echo "Temporarily close the bridge traffic and filter it through iptables"

sudo sysctl -w net.bridge.bridge-nf-call-iptables=0

echo "=== Bridge configuration completed ==="

```

Execution:

```bash
sudo chmod +x setup_bridge.sh
sudo bash setup_bridge.sh
```

---

## 3. Steps to allocate static IP on the side machine (the IP of the wired connection between the upper machine and the lower machine has generally been assigned, usually the 192.168.26.x network segment. Note that the network segment of the side machine should be in the same network segment)

### View existing network connection configuration

```bash
nmcli connection show
```

This command lists all current network connections with their name, UUID, type, and device.

### Modify the specified wired connection to a static IP configuration

Assume that the connection name to be modified is `"Wired Connection 1"`, configure the static IP address and mask as `192.168.26.10/24`, do not set a gateway, and configure it manually:

```bash
sudo nmcli connection modify "wired connection 1" ipv4.addresses 192.168.26.10/24 ipv4.gateway "" ipv4.method manual
```

Description:

- `ipv4.addresses` Set static IP address and subnet mask
- `ipv4.gateway` left blank means there is no default gateway
- `ipv4.method manual` is set to manual static IP configuration

### Activate modified network connection

```bash
sudo nmcli connection up "wired connection 1"
```

This command reactivates the specified connection to make the configuration take effect.

### Verify network configuration

Check the current interface IP:

```bash
ip addr show
```

Or view connection details:

```bash
nmcli connection show "wired connection 1"
```

After completing the above steps, the wired interface of the edge machine will use the static IP `192.168.26.10` to communicate within the corresponding subnet.

---

## 4. Verification steps

- Test pinging each other:

```bash
# Example
ping 192.168.26.12 # Side computer ping host computer
ping 192.168.26.10 # Host computer ping edge computer
```

- After pinging, set ROS_IP, ROS_MASTER_URI on the side machine:

```bash
# 1. Add a comment at the end of the ~/.bashrc file for easy identification
echo "# ROS network configuration" >> ~/.bashrc

# 2. Add ROS_IP environment variable
echo "export ROS_IP=192.168.26.10" >> ~/.bashrc

# 3. Add ROS_MASTER_URI environment variable
echo "export ROS_MASTER_URI=http://192.168.26.1:11311" >> ~/.bashrc

# 4. Make changes take effect immediately
source ~/.bashrc

```

- Test rostopic communication

```bash
# 2. Verify ros topic (no problem if there is data)
rostopic echo /sensors_data_raw
rostopic echo /cam_h/color/image_raw/compressed
rostopic echo /cam_r/color/image_raw/compressed
rostopic echo /cam_l/color/image_raw/compressed
rostopic echo /leju_claw_state # if you use leju_claw
rostopic echo /dexhand/state   # if you use qiangnao
```
