# Deploying phorse

A private beta on one DigitalOcean droplet, behind a site-wide password.

```
https://yourdomain  ──  Caddy  (HTTPS + the shared site password)
                         ├─ /api/*  →  gunicorn server.wsgi:app   (phorse-api.service)
                         └─ /       →  web/build/  (static site)
                        python -m server.scheduler                (phorse-scheduler.service)
                        deploy/backup.sh, nightly                 (phorse-backup.timer)
```

Everything runs on one machine because the API and the scheduler share
`server/data/` and `horse/data/head.pt` (see CLAUDE.md "Retrain loop").

Placeholders below: `YOUR_IP` is the droplet's IP address, `yourdomain.me`
is your domain.

---

## 0. Before you start

- **GitHub Student Developer Pack** (education.github.com/pack): claim the
  DigitalOcean credit and a free domain (Namecheap / name.com offers).
- **An SSH key** on your PC. In PowerShell: `ssh-keygen -t ed25519`, accept
  the defaults. The public half is `~\.ssh\id_ed25519.pub`.
- **This branch merged and pushed** to GitHub -- the server pulls from there.

## 1. Create the droplet

DigitalOcean → Create → Droplets:
- **Region:** London
- **Image:** Ubuntu 24.04 (LTS)
- **Size:** Basic → Regular → **4 GB RAM / 2 CPUs**. 2 GB is too tight
  (torch + CLIP in the API, plus the scheduler).
- **Authentication:** SSH key → add the contents of `id_ed25519.pub`.
- **Backups:** turn on weekly backups (about +20% -- the credit covers it).
  These are the only backups that survive losing the droplet.

Note the droplet's IP address.

## 2. Point the domain at it

At your registrar's DNS settings, add an **A record**: host `@`, value
`YOUR_IP`. Check it has taken effect (can take a few minutes to an hour):

```powershell
nslookup yourdomain.me
```

## 3. Base server setup

```bash
ssh root@YOUR_IP

apt update && apt upgrade -y
apt install -y git python3-venv python3-dev build-essential ufw

# Firewall: SSH + web only. The API listens on localhost, unreachable from outside.
ufw allow OpenSSH && ufw allow 80 && ufw allow 443 && ufw --force enable

# 2 GB swap: a safety net for the torch install and the first anchor build.
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

# Node 22 (Ubuntu's own nodejs is too old for Vite).
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt install -y nodejs

# Caddy (from caddyserver.com/docs/install -- check there if this fails).
apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' > /etc/apt/sources.list.d/caddy-stable.list
apt update && apt install -y caddy
```

## 4. App user + code

The app runs as its own unprivileged `phorse` user, from `/srv/phorse`.

```bash
useradd --system --create-home --home-dir /home/phorse --shell /bin/bash phorse
mkdir /srv/phorse && chown phorse:phorse /srv/phorse
```

**If the GitHub repo is public:**

```bash
sudo -u phorse -H git clone https://github.com/luke-wrathall/Phorse.git /srv/phorse
```

**If it's private**, give the server a read-only deploy key:

```bash
sudo -u phorse -H mkdir -p -m 700 /home/phorse/.ssh
sudo -u phorse -H ssh-keygen -t ed25519 -N "" -f /home/phorse/.ssh/id_ed25519
cat /home/phorse/.ssh/id_ed25519.pub
```

On GitHub: repo → Settings → Deploy keys → Add deploy key → paste it,
leave "Allow write access" off. Then:

```bash
sudo -u phorse -H git clone git@github.com:luke-wrathall/Phorse.git /srv/phorse
```

## 5. Anchor cache (optional, saves time)

Copying your local anchor cache skips the ~350 MB CIFAR-10 + PACS download
and re-embedding on the first retrain, and guarantees the same anchor set.
From PowerShell in the repo:

```powershell
scp horse/data/anchor_cache.npz root@YOUR_IP:/tmp/
```

Then on the server:

```bash
install -D -o phorse -g phorse /tmp/anchor_cache.npz /srv/phorse/horse/data/anchor_cache.npz
```

Don't copy `head.pt` or `app.db` -- the server starts with a fresh
anchors-only head and no accounts. Your local test data stays local.

## 6. Config, services, first build

```bash
mkdir -p /etc/phorse
cp /srv/phorse/deploy/phorse.env.example /etc/phorse/phorse.env
python3 -c "import secrets; print(secrets.token_hex(32))"   # copy this
nano /etc/phorse/phorse.env     # paste it as HORSE_SECRET_KEY, set your domain
chown root:phorse /etc/phorse/phorse.env && chmod 640 /etc/phorse/phorse.env

cp /srv/phorse/deploy/phorse-api.service /srv/phorse/deploy/phorse-scheduler.service \
   /srv/phorse/deploy/phorse-backup.service /srv/phorse/deploy/phorse-backup.timer \
   /etc/systemd/system/
systemctl daemon-reload
systemctl enable phorse-api phorse-scheduler
systemctl enable --now phorse-backup.timer

bash /srv/phorse/deploy/update.sh    # installs everything, builds the site, starts both services
```

The first run takes a while: torch, then CLIP (~600 MB) downloads on API
startup. Watch the scheduler do its first retrain:

```bash
journalctl -u phorse-scheduler -f     # Ctrl-C to stop watching
```

**Until it logs its first `retrain:` line, there's no `head.pt` and nothing
passes the gate** -- wait for that before testing uploads.

## 7. HTTPS + the site password

```bash
caddy hash-password                  # type the site password; copy the hash
cp /srv/phorse/deploy/Caddyfile /etc/caddy/Caddyfile
nano /etc/caddy/Caddyfile            # domain, username, hash (the REPLACE_ME lines)
mkdir -p /var/log/caddy && chown caddy:caddy /var/log/caddy
systemctl reload caddy
```

Open `https://yourdomain.me`: you should get a browser password prompt,
then the site. Make an account, upload, rate -- the whole loop.

Share the username + password with the people you invite. To change it,
re-run `caddy hash-password`, update the Caddyfile, `systemctl reload caddy`.

---

## Day to day

| task | command (on the server) |
|---|---|
| deploy new code (after pushing to GitHub) | `bash /srv/phorse/deploy/update.sh` |
| API logs | `journalctl -u phorse-api -f` |
| retrain logs | `journalctl -u phorse-scheduler -f` |
| web/HTTPS logs | `tail -f /var/log/caddy/phorse.log` |
| restart | `systemctl restart phorse-api phorse-scheduler` |
| change a knob (`HORSE_*`) | edit `/etc/phorse/phorse.env`, then restart |
| run a retrain now | `cd /srv/phorse && sudo -u phorse -H bash -c 'set -a; . /etc/phorse/phorse.env; .venv/bin/python -m server.retrain'` |
| list backups | `ls /var/backups/phorse` |
| copy backups to your PC (PowerShell) | `scp -r root@YOUR_IP:/var/backups/phorse ./phorse-backups` |

**Restoring a backup:** stop both services, copy `app.db` back to
`server/data/`, untar `uploads.tar.gz` into `server/data/`, copy `head.pt`
to `horse/data/`, `chown -R phorse:phorse /srv/phorse`, start the services.

**Credit:** the student credit runs out after a year (and the free domain
needs renewing). Set a billing alert in DigitalOcean so you're not surprised.
