from src.client.pb_trihyper import pb_trihyperClient
from src.server.trimoe_nofish_nofuzzy import trimoe_nofish_nofuzzyServer


class pb_trihyperServer(trimoe_nofish_nofuzzyServer):
    """Anonymous PB-TriHyper server alias.

    This wrapper preserves the existing implementation while exposing a
    reviewer-facing method name through FL-bench's dynamic method loader.
    """

    algorithm_name: str = "pb_trihyper"
    client_cls = pb_trihyperClient

    def __init__(self, args):
        # The inherited implementation reads args.trimoe_nofish_nofuzzy.
        # main.py also parses the parent method defaults for inherited servers,
        # so always point the inherited implementation at the reviewer-facing
        # pb_trihyper group to preserve the requested CLI/config values.
        args.trimoe_nofish_nofuzzy = args.pb_trihyper
        super().__init__(args)
