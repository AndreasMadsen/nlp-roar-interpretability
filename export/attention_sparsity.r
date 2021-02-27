library(tidyverse)

alpha.amount = list('babi-1', 'babi-2', 'babi-3', 'imdb', 'mimic-d', 'mimic-a', 'snli', 'sst') %>%
  map_dfr(function (dataset)
    read_csv(str_glue('../results/attention/{dataset}_s-0.csv.gz'), col_types = cols(
      split=col_factor(c('train', 'val', 'test')),
      observation=col_integer(),
      index=col_integer(),
      alpha=col_number()
    )) %>%
    mutate(dataset=dataset, seed=0) %>%
    group_by(dataset, seed, split, observation) %>%
    mutate(
      sorted.cumsum = cumsum(sort(alpha)),
    ) %>%
    summarise(
      total=n(),
      p80 = n() - sum(sorted.cumsum <= 0.20),
      p90 = n() - sum(sorted.cumsum <= 0.10),
      p95 = n() - sum(sorted.cumsum <= 0.05),
      p99 = n() - sum(sorted.cumsum <= 0.01)
    ) %>%
    pivot_longer(cols=c(p80, p90, p95, p99), names_to="percentage", values_to="amount")
  )

p = ggplot(alpha.amount %>% filter(split=='train'), aes(x = amount, fill=percentage)) +
  geom_histogram(aes(y = ..density..), position = "identity", alpha=0.5, bins=100) +
  facet_wrap(~ split + dataset + seed, scale='free') +
  labs(x = 'tokens attended to')
print(p)

alpha.stat = alpha.amount %>%
  group_by(dataset, seed, split, percentage) %>%
  summarise(
    amount=mean(amount)
  ) %>%
  group_by(dataset, split, percentage) %>%
  summarise(
    amount.mean=mean(amount),
    amount.ci=abs(qt(0.025, df=n()-1)*sd(amount)/sqrt(n()))
  )
