#!/usr/bin/env bash
# Run a phorse management command on the server, as the phorse user with
# its environment loaded. Run as root:
#
#   sudo bash /srv/phorse/deploy/manage.sh invites create --uses 300 --note "insta story"
#   sudo bash /srv/phorse/deploy/manage.sh invites list
#   sudo bash /srv/phorse/deploy/manage.sh admin posts
#   sudo bash /srv/phorse/deploy/manage.sh admin ban someone --purge
#   sudo bash /srv/phorse/deploy/manage.sh retrain
#   sudo bash /srv/phorse/deploy/manage.sh retrain --fresh   # after replacing the anchor set
#
# Each command's --help lists the rest.
set -euo pipefail

case "${1:-}" in
	invites | admin | retrain) ;;
	*)
		echo "usage: manage.sh invites|admin|retrain [args...]" >&2
		exit 2
		;;
esac

cd /srv/phorse
exec sudo -u phorse -H bash -c \
	'set -a; . /etc/phorse/phorse.env; set +a; exec .venv/bin/python -m "server.$0" "$@"' "$@"
