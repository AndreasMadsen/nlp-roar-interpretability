
.PHONY: sync sync-beluga sync-graham sync-cedar sync-beluga-cache sync-graham-cache sync-beluga-mimic download-beluga-results

default:
	echo "no default"

sync-beluga:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' \
		-e ssh ./ cc-beluga:~/workspace/nlproar

sync-graham:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' \
		-e ssh ./ cc-graham:~/workspace/nlproar

sync-cedar:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' --exclude='.git/' \
		-e ssh ./ cc-cedar:~/workspace/nlproar

sync-narval:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' --exclude='.git/' \
		-e ssh ./ cc-narval:~/workspace/nlproar

sync-beluga-cache:
	rsync --info=progress2 -urltv \
		-e ssh ./cache/ cc-beluga:~/scratch/nlproar/cache

sync-graham-cache:
	rsync --info=progress2 -urltv \
		-e ssh ./cache/ cc-graham:~/scratch/nlproar/cache

sync-cedar-mimic:
	rsync --info=progress2 -urltv \
		-e ssh ./mimic/ cc-cedar:~/scratch/nlproar/mimic

download-cedar-results:
	rsync --info=progress2 -urltv \
		-e ssh cc-cedar:~/scratch/nlproar/results/ ./results

schedule-base:
	bash batch_jobs/babi.sh
	bash batch_jobs/imdb.sh
	bash batch_jobs/mimic.sh
	bash batch_jobs/stanford_nli.sh
	bash batch_jobs/stanford_sentiment.sh

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
