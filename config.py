def xsum_setting(args):
    # default setting for xsum
    args.dataset = getattr(args, 'dataset', 'xsum')
    args.batch_size = getattr(args, 'batch_size', 4)
    args.epoch = getattr(args, 'epoch', 5)      
    args.report_freq = getattr(args, 'report_freq', 100)
    args.eval_interval = getattr(args, 'eval_interval', 1000)   
    args.accumulate_step = getattr(args, 'accumulate_step', 12)
    args.model_type = getattr(args, 'model_type', 'roberta-base') 
    args.max_lr = getattr(args, 'max_lr', 2e-3)     
    args.seed = getattr(args, 'seed', 970903)
    args.datatype = getattr(args, 'datatype', 'with_neg_random')    # diverse
    args.max_len = getattr(args, 'max_len', 80)     # max length of summary
    args.max_num = getattr(args, 'max_num', 16)     # max number of candidate summaries
    args.total_len = getattr(args, 'total_len', 512)    # total length of source article
    args.gen_max_len = getattr(args, 'gen_max_len', 62)     # max length of generated summaries
    args.gen_min_len = getattr(args, 'gen_min_len', 11)
    args.grad_norm = getattr(args, 'grad_norm', 0)
    args.pretrained = getattr(args, 'pretrained', None)
    args.warmup_steps = getattr(args, "warmup_steps", 10000)
    args.margin = getattr(args, 'margin', 0.1)     
    args.temp = getattr(args, 'temp', 0.05)
    args.rank_scale = getattr(args, 'rank_scale', 10)       
    args.nll_scale = getattr(args, 'nll_scale', 0.1)         
    args.neg_size = getattr(args, 'neg_size', 4)
    args.thre = getattr(args, 'thre', 0.85)
    args.is_IW = getattr(args, 'is_IW', True)


def cnndm_setting(args):
    # default setting for cnndm
    args.dataset = getattr(args, 'dataset', 'cnndm')
    args.batch_size = getattr(args, 'batch_size', 4)   
    args.epoch = getattr(args, 'epoch', 5)
    args.report_freq = getattr(args, 'report_freq', 100)
    args.eval_interval = getattr(args, 'eval_interval', 1000)    
    args.accumulate_step = getattr(args, 'accumulate_step', 12)
    args.model_type = getattr(args, 'model_type', 'roberta-base')
    args.max_lr = getattr(args, 'max_lr', 2e-3)     
    args.seed = getattr(args, 'seed', 970903)
    args.datatype = getattr(args, 'datatype', 'with_neg_random')    # diverse
    args.max_len = getattr(args, 'max_len', 120)     # max length of summary
    args.max_num = getattr(args, 'max_num', 16)     # max number of candidate summaries
    args.total_len = getattr(args, 'total_len', 512)  # Roberta both max position is 512 
    args.gen_max_len = getattr(args, 'gen_max_len', 140)     # max length of generated summaries
    args.gen_min_len = getattr(args, 'gen_min_len', 55)
    args.grad_norm = getattr(args, 'grad_norm', 0)
    args.pretrained = getattr(args, 'pretrained', None)
    args.warmup_steps = getattr(args, "warmup_steps", 10000)
    args.margin = getattr(args, 'margin', 1)      
    args.temp = getattr(args, 'temp', 0.05)
    args.rank_scale = getattr(args, 'rank_scale', 10)      
    args.nll_scale = getattr(args, 'nll_scale', 0.1)     
    args.neg_size = getattr(args, 'neg_size', 4)
    args.thre = getattr(args, 'thre', 0.9)
    args.is_IW = getattr(args, 'is_IW', True)
