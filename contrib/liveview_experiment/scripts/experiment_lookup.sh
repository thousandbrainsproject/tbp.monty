#!/bin/bash
# Finding the experiment a person named.
#
# Monty's `conf/experiment` is a tree, not a directory: `evidence` lives
# at `test/evidence_lm/evidence`, and that whole path — not the file name
# — is the name Hydra knows it by. This resolves what somebody types into the
# one config it means, and says which tree it came from, because a config in
# `contrib/liveview_experiment/conf` is already a LiveView experiment and one in
# `src/tbp/monty/conf` is not.
#
# Sourced by `run.sh` and `run_parallel.sh` so the two answer alike.

# Set by the caller, before sourcing:
#   LIVEVIEW_CONF  the contrib tree, `contrib/liveview_experiment/conf/experiment`
#   MONTY_CONF     Monty's own tree, `src/tbp/monty/conf/experiment`

# Every experiment config there is, as "<source> <name>", the contrib tree first.
# A name that both trees have is the contrib one: it is the LiveView variant of
# the same experiment, and it is what a person asking for that name wants.
experiment_configs() {
    {
        if [ -d "$LIVEVIEW_CONF" ]; then
            (cd "$LIVEVIEW_CONF" && find . -type f -name '*.yaml') |
                sed 's|^\./||; s|\.yaml$||' |
                awk '{ print "contrib " $0 }'
        fi
        if [ -d "$MONTY_CONF" ]; then
            (cd "$MONTY_CONF" && find . -type f -name '*.yaml') |
                sed 's|^\./||; s|\.yaml$||' |
                awk '{ print "main " $0 }'
        fi
    } | awk '!seen[$2]++'
}

# The config a name means, as "<source> <name>".
#
#   exit 0  exactly one config; it is printed
#   exit 1  the name fits more than one config; they are printed for the caller
#   exit 2  nothing fits
resolve_experiment() {
    local name="$1" exact fit count

    exact=$(experiment_configs | awk -v want="$name" '$2 == want')
    if [ -n "$exact" ]; then
        printf '%s\n' "$exact"
        return 0
    fi

    # A bare file name is how people say a nested experiment — `evidence`
    # for `test/evidence_lm/evidence` — so it is answered when it is
    # unambiguous and refused when it is not: picking one of two configs with the
    # same name would be running an experiment nobody asked for.
    fit=$(experiment_configs | awk -v want="$name" '{ n = split($2, parts, "/"); if (parts[n] == want) print }')

    count=$(printf '%s' "$fit" | grep -c . 2>/dev/null || true)
    if [ "$count" -gt 1 ]; then
        printf '%s\n' "$fit"
        return 1
    fi
    if [ "$count" -eq 1 ]; then
        printf '%s\n' "$fit"
        return 0
    fi

    return 2
}
