#!/bin/bash

#Bridge interface name
BRIDGE=br0

#Physical network card name (replace with your interface name)
IFACE1=enx00e04c684355
IFACE2=enxc8a362b260f5

#Check if the bridge interface exists
if ip link show "$BRIDGE" &>/dev/null; then
    echo "Error: Bridge interface $BRIDGE already exists, please delete it first or use another name."
    exit 1
fi

#Check if the physical network card exists
if ! ip link show "$IFACE1" &>/dev/null; then
    echo "Error: Physical network card $IFACE1 does not exist, exit script."
    exit 1
fi

if ! ip link show "$IFACE2" &>/dev/null; then
    echo "Error: Physical network card $IFACE2 does not exist, exit script."
    exit 1
fi

#Bridge IP address
BRIDGE_IP=192.168.26.1/24

echo "=== Deactivate interface ==="
sudo ip link set dev $IFACE1 down
sudo ip link set dev $IFACE2 down

echo "=== Clear interface IP address ==="
sudo ip addr flush dev $IFACE1
sudo ip addr flush dev $IFACE2

echo "=== Create bridge interface ==="
sudo ip link add name $BRIDGE type bridge

echo "=== Add the physical interface to the bridge ==="
sudo ip link set dev $IFACE1 master $BRIDGE
sudo ip link set dev $IFACE2 master $BRIDGE

echo "=== Start physical interface and bridge interface ==="
sudo ip link set dev $IFACE1 up
sudo ip link set dev $IFACE2 up
sudo ip link set dev $BRIDGE up

echo "=== Assign an IP address to the bridge interface ==="
sudo ip addr add $BRIDGE_IP dev $BRIDGE

echo "=== Show interface status ==="
ip addr show $BRIDGE
ip addr show $IFACE1
ip addr show $IFACE2

echo "Temporarily shut down bridge traffic through iptables filtering"

sudo sysctl -w net.bridge.bridge-nf-call-iptables=0

echo "=== Bridge configuration completed ==="
