# Microscope LAN — setup & runbook

How we wired two microscope-control PCs together over a dumb switch with no
DHCP, made them reachable by name across reboots, and how to replicate it for a
fresh network or add another machine.

---

## 1. Motivation

Each Cephla SQUID microscope is driven by its own PC running seafront. We want
those PCs on a **small, private, wired network** so they can:

- be reached over SSH for admin/headless control, and
- (later) let seafront instances talk to each other directly.

The hardware is just: each PC's wired NIC → a **plain unmanaged switch**. No
router, no DHCP server, no internet on that segment. Each PC keeps its *normal*
internet connection on a separate interface (here: Wi‑Fi / `wlan0` on the
university network); the wired link is a dedicated island.

This is deliberately isolated — it keeps microscope traffic off the campus
network and means the setup works the same anywhere you plug the switch in.

---

## 2. The core problem (why "just plug it in" didn't work)

On a segment with **no DHCP server**, default configs fail in two ways:

1. **NetworkManager DHCP-retry loop.** The default wired profile uses
   `ipv4.method = auto` (DHCP). With no DHCP server it retries forever, sitting
   in state `connecting (getting IP configuration)`, and eventually **drops the
   interface and strips its address**. Symptom: the device shows `disconnected`
   with no IP, and is invisible on the wire even though the cable link is up.
   Each retry cycle also flushes the neighbor cache, so any connectivity is
   intermittent (works for a few seconds, then 100% packet loss).

2. **No agreed addressing.** With no DHCP and no static plan, there is nothing
   to address packets to.

**The fix for both:** tell NetworkManager to use **link-local addressing**
(`ipv4.method = link-local`) instead of chasing DHCP. The interface then comes
up instantly and *stays* up with:

- an IPv4 link-local address `169.254.x.x/16` (zeroconf / APIPA), and
- an IPv6 link-local address `fe80::.../64`.

Both ends self-assign with zero configuration and zero servers.

### Names, not addresses (the reboot-proofing)

Link-local addresses are **not stable**: the IPv4 `169.254.x.x` is randomly
re-picked every boot, and the IPv6 `fe80::` (stable-privacy) can change too. So
we never hard-code an address. Instead each machine runs **Avahi (mDNS)**, which
advertises `<hostname>.local` and re-publishes whatever address it currently
holds. `ssh user@host.local` then works across reboots regardless of the
underlying address.

For mDNS resolution to work through normal tools (`ssh`, `ping`, `getent`), the
host's `/etc/nsswitch.conf` `hosts:` line must include the nss-mdns plugin —
installing the package alone is not enough on Arch (Ubuntu's package wires it up
automatically).

---

## 3. Current network (as built)

| | This machine | Peer |
|---|---|---|
| Hostname | `workstation` | `lab3` |
| Login user | _(yours)_ | `pharmbio` |
| OS | CachyOS (Arch-based) | Ubuntu 24.04 |
| Net manager | NetworkManager | NetworkManager (Variant A) |
| Wired iface | `eth0` (USB adapter; altnames `enp197s0f3u1u2c2`, `enx0826ae3eaac0`) | `enp2s0` |
| Wired MAC | `08:26:ae:3e:aa:c0` | `a0:36:bc:58:71:4c` |
| Wired link-local | `169.254.28.162/16`, `fe80::781a:99b0:b413:fb27` | `fe80::f5ec:9ac3:ff49:4f3e` |
| Internet (separate) | `wlan0` (eduroam) | its own normal connection |

Physical: `workstation eth0` → switch ← `lab3 enp2s0`. Link negotiated at 100 Mbps.

Login: `ssh pharmbio@lab3.local`  (resolves via mDNS to whatever address `enp2s0`
currently holds — reboot-stable by name).

---

## 4. Setup — this machine (`workstation`, Arch/CachyOS + NetworkManager)

Already applied. Recorded for replication.

```bash
# 1. Switch the wired profile from DHCP to link-local, persistent across reboots.
#    ("Wired connection 1" is NetworkManager's auto-created profile name; check
#     yours with:  nmcli -t -f NAME,DEVICE connection show )
sudo nmcli connection modify "Wired connection 1" \
     ipv4.method link-local ipv6.method link-local connection.autoconnect yes
sudo nmcli connection up "Wired connection 1"

# 2. Open the firewall on the trusted wired link only (internet stays firewalled
#    on wlan0). ufw is active on this machine.
sudo ufw allow in on eth0 comment 'microscope LAN'

# 3. Enable mDNS name resolution through glibc (nss-mdns is installed but Arch
#    does NOT wire it into nsswitch automatically). Inserts the plugin into the
#    hosts: line if not already present.
grep -q mdns /etc/nsswitch.conf || sudo sed -i \
   's/^hosts: mymachines /hosts: mymachines mdns_minimal [NOTFOUND=return] /' \
   /etc/nsswitch.conf
# Resulting line:
# hosts: mymachines mdns_minimal [NOTFOUND=return] resolve [!UNAVAIL=return] files myhostname dns

# Avahi was already installed and running (systemctl is-active avahi-daemon).
# If not: sudo pacman -S --needed avahi nss-mdns && sudo systemctl enable --now avahi-daemon
```

Verify:

```bash
nmcli -t -f DEVICE,STATE,CONNECTION device | grep eth0   # -> eth0:connected:...
ip -br addr show eth0                                     # -> 169.254.x.x + fe80::
getent hosts workstation.local                           # -> resolves (mDNS works)
```

---

## 5. Setup — peer (Ubuntu 24.04)

> **Package names differ from Arch.** Ubuntu uses `openssh-server`,
> `avahi-daemon`, `libnss-mdns` (not `openssh` / `avahi` / `nss-mdns`).
> Ubuntu's `libnss-mdns` package **auto-edits `nsswitch.conf`** — no manual sed.
> apt needs internet, which the isolated switch does not provide — install over
> the peer's normal connection (Wi‑Fi / its other LAN).

### 5a. Install services

```bash
sudo apt update
sudo apt install -y openssh-server avahi-daemon libnss-mdns
sudo systemctl enable --now ssh avahi-daemon   # Ubuntu's unit is "ssh", not "sshd"
```

### 5b. Configure the wired interface as link-local (pick the matching variant)

First identify the manager and interface:

```bash
systemctl is-active NetworkManager systemd-networkd
ip -br link | grep -ivE 'lo|wl|docker|veth|br-|tun'      # the wired iface name
```

**Variant A — NetworkManager (Ubuntu Desktop):**

```bash
IFACE=<wired-iface>
CON=$(nmcli -t -f NAME,DEVICE connection show | awk -F: -v i="$IFACE" '$2==i{print $1; exit}')
CON=${CON:-"Wired connection 1"}
sudo nmcli connection modify "$CON" \
     ipv4.method link-local ipv6.method link-local connection.autoconnect yes
sudo nmcli connection up "$CON"
```

**Variant B — netplan + systemd-networkd (Ubuntu Server):** create
`/etc/netplan/60-microscope-lan.yaml` (replace `<wired-iface>`):

```yaml
network:
  version: 2
  ethernets:
    <wired-iface>:
      dhcp4: false
      dhcp6: false
      link-local: [ipv4, ipv6]   # self-assign 169.254.x.x + fe80::
      optional: true             # don't block boot waiting for this link
```

```bash
sudo chmod 600 /etc/netplan/60-microscope-lan.yaml
sudo netplan apply
```

### 5c. Firewall (only if ufw is active on the peer)

```bash
IFACE=<wired-iface>
if systemctl is-active --quiet ufw; then
  sudo ufw allow in on "$IFACE" to any port 22 proto tcp comment 'ssh microlan'
  sudo ufw allow in on "$IFACE" to any port 5353 proto udp comment 'mdns microlan'
fi
```

### 5d. Report identity (needed to connect)

```bash
echo "PEER HOSTNAME: $(hostname)"
echo "PEER USER:     $(whoami)"
ip -br addr show <wired-iface>
```

---

## 6. Connecting

From any machine on the LAN:

```bash
ssh <user>@<peer-hostname>.local
```

For passwordless login (do this once, after the first successful password login):

```bash
ssh-copy-id <user>@<peer-hostname>.local
```

---

## 7. Adding another computer to this network

The network is fully decentralized — no central server, every node is equal.
To add machine N:

1. Plug its wired NIC into the switch.
2. Give it a **unique hostname** (`sudo hostnamectl set-hostname <name>`) — mDNS
   names collide otherwise.
3. Run the setup for its OS:
   - Arch/CachyOS → section 4.
   - Ubuntu → section 5.
   - (General rule for any Linux: wired iface = link-local + autoconnect;
     install & enable `sshd` + `avahi`; ensure nss-mdns is in `nsswitch.conf`;
     open the firewall on the wired iface.)
4. From an existing node, confirm: `getent hosts <name>.local` then
   `ssh <user>@<name>.local`.

No address bookkeeping is required — Avahi handles discovery, and addresses are
self-assigned. The switch just needs enough ports.

---

## 8. Replicating on a brand-new network

Same as above with nothing pre-existing:

1. Unmanaged switch, no router/DHCP needed.
2. Each PC: wired NIC → switch; keep internet on a separate interface.
3. Apply the per-OS setup (sections 4 / 5) on every node, each with a unique
   hostname.
4. Done — nodes find each other as `<hostname>.local`.

If you ever outgrow link-local (e.g. want stable, routable addresses or to add a
router), the alternative is static IPs on a private subnet (e.g. `192.168.77.0/24`,
`.1`, `.2`, …) — more bookkeeping, but addresses don't change. Link-local + mDNS
was chosen here because it is zero-config and self-healing.

---

## 9. Discovery & troubleshooting cheat-sheet

All run from a working node, on the wired interface (`eth0` here):

```bash
# Who else is on the wire? (IPv6 all-nodes multicast — every live host replies)
ping -6 -c 5 ff02::1%eth0
#   -> you'll see your own fe80:: plus one line per neighbor. Note their fe80::.

# Neighbor / ARP tables (after some traffic)
ip -6 neigh show dev eth0          # IPv6 neighbors + MACs
ip -4 neigh show dev eth0          # IPv4 (169.254.x.x) neighbors

# Populate the IPv4 ARP table by pinging the link-local broadcast
ping -c 3 -b 169.254.255.255 -I eth0 ; ip -4 neigh show dev eth0

# Browse advertised mDNS services / resolve a name
avahi-browse -art
getent hosts <host>.local
avahi-resolve -n <host>.local

# Is a service up on a peer? (bash builtin, no nmap needed)
timeout 2 bash -c 'echo > /dev/tcp/<host>.local/22' && echo OPEN || echo closed

# Interface health
nmcli -t -f DEVICE,STATE,CONNECTION device          # NM state
ip -br addr show eth0                                # addresses present?
cat /sys/class/net/eth0/carrier                      # 1 = cable link up
```

### Common symptoms

| Symptom | Cause | Fix |
|---|---|---|
| `eth0` shows `disconnected`, no IP | NM stuck in DHCP retry, dropped the device | set `ipv4.method link-local` (sec. 4/5) |
| Intermittent ping (works, then 100% loss) | NM DHCP-retry loop flushing neighbor cache | same as above |
| Peer pings but `ssh host.local` says name not found | nss-mdns not in `nsswitch.conf` | add `mdns_minimal [NOTFOUND=return]` (sec. 4 step 3) |
| Pings work but port 22 closed | `sshd` not installed/enabled, or firewall | install/enable `ssh`; `ufw allow in on <iface>` |
| Peer invisible entirely | peer's wired iface down/no address, or powered off | fix peer per sec. 5; check its `ip -br addr` |
| IPv6 link-local ping needs `%eth0` | link-local addresses require a scope/zone id | always append `%<iface>` for `fe80::` targets |

---

## 10. Notes / gotchas

- **USB Ethernet adapter naming.** The wired NIC on `workstation` is a USB
  adapter; its kernel name (`eth0` vs `enp...u...`) can vary by port. The
  NetworkManager profile binds to it, but if you move it to a different USB port
  and the name changes, re-check `nmcli device`.
- **`fe80::` always needs a scope.** Link-local IPv6 targets must be written as
  `fe80::...%eth0`. mDNS/`getent` handle this for you; raw addresses don't.
- **Firewall scope.** We open the firewall only *on the wired interface*, so the
  internet-facing interface (`wlan0`) stays protected.
- **Internet on the isolated segment.** There is none by design. Install
  packages over each machine's separate internet connection before/while it's
  also on the switch.
- **Ubuntu service name.** The SSH unit on Ubuntu is `ssh` (`systemctl ... ssh`),
  not `sshd` as on Arch.

---

## 11. Remote access to seafront (the actual goal)

SSH was only the bring-up/proof. The real point is to reach a microscope's
**seafront web UI/API** running on the remote PC (`lab3`) from another machine
(`workstation`) over the wired LAN.

### The bind-address fix

seafront historically bound its HTTP server to `127.0.0.1` (loopback only), so it
was unreachable from other machines. The bind address is now configurable
(`seafront/config/basics.py` `ServerConfig.host`, plus a `--host` CLI flag in
`seafront/__main__.py`). **Default is still `127.0.0.1`** — exposing on the
network is opt-in.

- `--host 127.0.0.1` (default) — local only.
- `--host 0.0.0.0` — all IPv4 interfaces (IPv4 only — `fe80::` would be unreachable).
- `--host ::` — **use this.** seafront binds it as a true **dual-stack** socket
  (it sets `IPV6_V6ONLY=0`; plain uvicorn would otherwise make `::` IPv6-only),
  so the server answers on *both* the IPv6 `fe80::` and the IPv4 `169.254.x.x`
  link-local addresses. That matters because `lab3.local` resolves to both and a
  browser may pick either. Verified: `http://127.0.0.1`, `http://169.254.x.x`,
  and `http://[fe80::…%iface]` all return 200 against a `--host ::` server.

### Run seafront on the microscope PC (`lab3`)

```bash
# pick any free port (SSH owns 22; use e.g. 5000/8000)
uv run python -m seafront --microscope "squid" --host :: --port 8000
#   or set host/port permanently in ~/seafront/config.json
```

### Open the firewall for that port — on the wired link only

```bash
sudo ufw allow in on enp2s0 to any port 8000 proto tcp comment 'seafront'
```

> **Security:** binding to `::` listens on *all* interfaces including `lab3`'s
> internet-facing Wi‑Fi. The `ufw allow in on enp2s0` rule opens the port **only
> on the microscope LAN** — ufw's default-deny keeps it closed on every other
> interface. Keep ufw enabled; that's what restricts exposure to the private link.

### Connect from `workstation`

```
http://lab3.local:8000          # preferred — reboot-stable name (mDNS)
http://169.254.86.162:8000      # fallback — current IPv4 link-local (changes on reboot)
```

WebSockets work over the same host/port (same origin), so live image streaming
and control work unchanged.

### Deploying to lab3

The bind-address fix is committed. On `lab3`: `git pull` its seafront checkout
(or `scp` the two files `seafront/config/basics.py`, `seafront/__main__.py` over
the SSH link). Confirmed working end-to-end: `http://lab3.local:8000` from
`workstation`.

---

## 12. Planned: gateway PC (central access point)

Scaling from the 2-machine link to the full lab: a **5-port switch** with **4
microscope PCs** (each running seafront, driving one SQUID over USB) plus a
**5th PC as the gateway**. The gateway hosts a Wi-Fi hotspot; you join it from a
laptop/phone, open one dashboard, and reach every microscope through it. The
microscope PCs are never contacted directly by clients.

```
        Wi-Fi hotspot ── clients (laptop/phone) join here
              │
        ┌─────┴──────┐
        │ Gateway PC │  :8000  FastAPI dashboard
        │            │  :8001→squid1:8000  …  :8004→squid4:8000  (Caddy)
        └─────┬──────┘
          5-port switch  (wired backbone)
        ┌────┬────┬────┬────┐
       PC1  PC2  PC3  PC4      each: seafront --host :: --port 8000
```

### Components on the gateway

1. **Wi-Fi hotspot** — `nmcli device wifi hotspot` (NetworkManager runs DHCP for
   clients). Clients only ever talk to the gateway.
2. **Caddy reverse proxy** — one port per microscope, forwarded **root→root** so
   seafront's HTML, API, and WebSocket all work with no URL rewriting. WebSocket
   upgrade is automatic. Minimal `Caddyfile`:
   ```
   :8001 { reverse_proxy squid1:8000 }
   :8002 { reverse_proxy squid2:8000 }
   :8003 { reverse_proxy squid3:8000 }
   :8004 { reverse_proxy squid4:8000 }
   ```
3. **FastAPI dashboard** (`:8000`) — a separate app serving the landing page. It
   queries each microscope's seafront API **on demand** (status, and links to its
   `:800N` port) and renders a card per microscope. Runs alongside Caddy.

### Backbone addressing

For this permanent, always-on setup, **static IPs on the wired NICs** are
recommended over link-local + mDNS: stable proxy upstreams, no per-boot address
changes, and no reliance on `.local` resolution inside Caddy/Go (whose pure-Go
resolver may bypass nss-mdns). E.g. gateway `192.168.50.1`, squid1–4
`192.168.50.11`–`.14`; point the `Caddyfile` at those addresses. Link-local +
mDNS (sections 4–5) remains the zero-config option for ad-hoc bring-up.

### Status

Design agreed (Caddy forwarding + parallel FastAPI dashboard). Not yet built.
```
