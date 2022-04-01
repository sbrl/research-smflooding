import clip

# @source https://stackoverflow.com/a/67583699/1460422
def do_collate_filter_none(dataset, batch):
    len_batch = len(batch)
    batch = list(filter(lambda x: x is not None, batch))

    if len_batch > len(batch):                
        db_len = len(dataset)
        diff = len_batch - len(batch)
        while diff != 0:
            a = dataset[np.random.randint(0, db_len)]
            if a is None:                
                continue
            batch.append(a)
            diff -= 1

    return torch.utils.data.dataloader.default_collate(batch)
