
.PHONY: sync sync-beluga sync-cedar sync-beluga-cache sync-beluga-mimic download-beluga-results

sync: sync-beluga

sync-beluga:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' \
		-e ssh ./ cc-beluga:~/workspace/comp550

sync-cedar:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' \
		-e ssh ./ cc-cedar:~/workspace/comp550

sync-beluga-cache:
	rsync --info=progress2 -urltv \
		-e ssh ./cache/ cc-beluga:~/scratch/comp550/cache

sync-beluga-mimic:
	rsync --info=progress2 -urltv \
		-e ssh ./mimic/ cc-beluga:~/scratch/comp550/mimic

download-beluga-results:
	rsync --info=progress2 -urltv --delete \
		-e ssh cc-beluga:~/scratch/comp550/results/ ./results

schedule-roar:
	bash batch_jobs/babi_roar.sh
	bash batch_jobs/imdb_roar.sh
	bash batch_jobs/mimic_roar.sh
	bash batch_jobs/stanford_nli_roar.sh
	bash batch_jobs/stanford_sentiment_roar.sh

schedule-roar-recursive:
	bash batch_jobs/babi_roar_recursive.sh
	bash batch_jobs/imdb_roar_recursive.sh
	bash batch_jobs/mimic_roar_recursive.sh
	bash batch_jobs/stanford_nli_roar_recursive.sh
	bash batch_jobs/stanford_sentiment_roar_recursive.sh
