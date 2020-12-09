
.PHONY: sync sync-cache download-results

sync:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' \
		-e ssh ./ cc-beluga:~/workspace/comp550

sync-cache:
	rsync --info=progress2 -urltv \
		-e ssh ./cache/ cc-beluga:~/scratch/comp550/cache

download-results:
	rsync --info=progress2 -urltv --delete \
		-e ssh cc-beluga:~/scratch/comp550/results/ ./results
