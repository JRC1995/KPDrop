import importlib


def load_config(args):

    config_module = importlib.import_module("configs.{}_configs".format(args.dataset))
    config = getattr(config_module, "{}_config".format(args.model))
    config_obj = config()
    config_dict = {}
    obj_attributes = [attribute for attribute in dir(config_obj) if not attribute.startswith('__')]
    for attribute in obj_attributes:
        config_dict[attribute] = eval("config_obj.{}".format(attribute))
    config_dict["dataset"] = args.dataset
    config_dict["model_type"] = args.model_type

    if args.decode_mode == "Greedy":
        config_dict["hard_exclusion"] = False
        config_dict["beam_search"] = False
        config_dict["rerank"] = False
    elif args.decode_mode == "GreedyES1":
        config_dict["hard_exclusion"] = True
        config_dict["ex_window_size"] = 1
        config_dict["beam_search"] = False
        config_dict["rerank"] = False
    elif args.decode_mode == "GreedyES4":
        config_dict["hard_exclusion"] = True
        config_dict["ex_window_size"] = 4
        config_dict["beam_search"] = False
        config_dict["rerank"] = False
    elif args.decode_mode == "Beam":
        config_dict["hard_exclusion"] = False
        config_dict["beam_search"] = True
        config_dict["length_normalization"] = False
        config_dict["rerank"] = False
        if config_dict["one2one"]:
            config_dict["max_beam_size"] = 200
            config_dict["beam_width"] = 200
        else:
            config_dict["max_beam_size"] = 50
            config_dict["beam_width"] = 50

        config_dict["beam_threshold"] = 0.0
    elif args.decode_mode == "BeamLN":
        config_dict["hard_exclusion"] = False
        config_dict["beam_search"] = True
        config_dict["length_normalization"] = True
        config_dict["rerank"] = False
        if config_dict["one2one"]:
            config_dict["max_beam_size"] = 200
            config_dict["beam_width"] = 200
        else:
            config_dict["max_beam_size"] = 50
            config_dict["beam_width"] = 50
        config_dict["beam_threshold"] = 0.0
    elif args.decode_mode == "Beam20LN":
        config_dict["hard_exclusion"] = False
        config_dict["beam_search"] = True
        config_dict["length_normalization"] = True
        config_dict["beam_width"] = 20
        config_dict["rerank"] = False
        config_dict["max_beam_size"] = 20
        config_dict["beam_threshold"] = 0.0
    elif args.decode_mode == "Beam50LN":
        config_dict["hard_exclusion"] = False
        config_dict["beam_search"] = True
        config_dict["length_normalization"] = True
        config_dict["beam_width"] = 50
        config_dict["rerank"] = False
        config_dict["max_beam_size"] = 50
        config_dict["beam_threshold"] = 0.0
    elif args.decode_mode == "Beam11LN":
        config_dict["hard_exclusion"] = False
        config_dict["beam_search"] = True
        config_dict["length_normalization"] = True
        config_dict["beam_width"] = 11
        config_dict["rerank"] = False
        config_dict["max_beam_size"] = 11
        config_dict["beam_threshold"] = 0.0
    elif args.decode_mode == "TopBeam5LN":
        config_dict["hard_exclusion"] = False
        config_dict["beam_search"] = True
        config_dict["length_normalization"] = True
        config_dict["beam_width"] = 5
        config_dict["top_beam"] = True
        config_dict["rerank"] = False
        config_dict["max_beam_size"] = 5
        config_dict["beam_threshold"] = 0.0
    elif args.decode_mode == "TopBeam50LN":
        config_dict["hard_exclusion"] = False
        config_dict["beam_search"] = True
        config_dict["length_normalization"] = True
        config_dict["top_beam"] = True
        config_dict["rerank"] = False
        if config_dict["one2one"]:
            config_dict["max_beam_size"] = 200
            config_dict["beam_width"] = 50
        else:
            config_dict["max_beam_size"] = 50
            config_dict["beam_width"] = 50
        config_dict["beam_threshold"] = 0.0
    elif args.decode_mode == "AdaBeam50LN":
        config_dict["hard_exclusion"] = False
        config_dict["beam_search"] = True
        config_dict["length_normalization"] = True
        config_dict["beam_width"] = 50
        config_dict["rerank"] = False
        config_dict["max_beam_size"] = 50
        config_dict["beam_threshold"] = 0.015
    elif args.decode_mode == "AdaBeam20LN":
        config_dict["dev_batch_size"] = 12
        config_dict["batch_size"] = 12
        config_dict["hard_exclusion"] = False
        config_dict["beam_search"] = True
        config_dict["length_normalization"] = True
        config_dict["beam_width"] = 20
        config_dict["max_beam_size"] = 20
        config_dict["rerank"] = False
        config_dict["beam_threshold"] = 0.015
    elif args.decode_mode == "BeamLN_ES4":
        config_dict["hard_exclusion"] = True
        config_dict["ex_window_size"] = 4
        config_dict["beam_search"] = True
        config_dict["length_normalization"] = True
        config_dict["beam_width"] = 50
        config_dict["rerank"] = False
        config_dict["max_beam_size"] = 50
        config_dict["beam_threshold"] = 0.0
    elif args.decode_mode == "BeamLN_ES1":
        config_dict["hard_exclusion"] = True
        config_dict["ex_window_size"] = 1
        config_dict["beam_search"] = True
        config_dict["length_normalization"] = True
        config_dict["beam_width"] = 50
        config_dict["rerank"] = False
        config_dict["max_beam_size"] = 50
        config_dict["beam_threshold"] = 0.0




    return config_dict
