
.PHONY: sync sync-cache

sync:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' \
		-e ssh ./ cc-beluga:~/workspace/comp550

sync-cache:
	rsync --info=progress2 -urltv \
		-e ssh ./cache/ cc-beluga:~/scratch/comp550/cache

