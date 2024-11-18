# -----------------------------------------------------------------------------
# ModelFactory
# -----------------------------------------------------------------------------
class ModelFactory:
    # Class-level registries
    backbones = {}
    decoders = {}
    heads = {}

    # -------------------------------------------------------------------------
    # register_backbone
    # -------------------------------------------------------------------------
    @classmethod
    def register_backbone(cls, name: str, backbone_cls):
        """Register a new backbone in the factory."""
        cls.backbones[name] = backbone_cls

    # -------------------------------------------------------------------------
    # register_decoder
    # -------------------------------------------------------------------------
    @classmethod
    def register_decoder(cls, name: str, decoder_cls):
        """Register a new decoder in the factory."""
        cls.decoders[name] = decoder_cls

    # -------------------------------------------------------------------------
    # register_head
    # -------------------------------------------------------------------------
    @classmethod
    def register_head(cls, name: str, head_cls):
        """Register a new head in the factory."""
        cls.heads[name] = head_cls

    # -------------------------------------------------------------------------
    # get_component
    # -------------------------------------------------------------------------
    @classmethod
    def get_component(cls, component_type: str, name: str, **kwargs):
        """Public method to retrieve and instantiate a component by type and name."""  # noqa: E501
        print(cls.backbones)
        print(cls.decoders)
        print(cls.heads)
        registry = {
            "encoder": cls.backbones,
            "decoder": cls.decoders,
            "head": cls.heads,
        }.get(component_type)

        if registry is None or name not in registry:
            raise ValueError(f"{component_type.capitalize()} '{name}' not found in registry.")  # noqa: E501

        return registry[name](**kwargs)

    # -------------------------------------------------------------------------
    # encoder
    # -------------------------------------------------------------------------
    @classmethod
    def encoder(cls, name):
        """Class decorator for registering an encoder."""
        def decorator(encoder_cls):
            cls.register_backbone(name, encoder_cls)
            return encoder_cls
        return decorator

    # -------------------------------------------------------------------------
    # decoder
    # -------------------------------------------------------------------------
    @classmethod
    def decoder(cls, name):
        """Class decorator for registering a decoder."""
        def decorator(decoder_cls):
            cls.register_decoder(name, decoder_cls)
            return decoder_cls
        return decorator

    # -------------------------------------------------------------------------
    # head
    # -------------------------------------------------------------------------
    @classmethod
    def head(cls, name):
        """Class decorator for registering a head."""
        def decorator(head_cls):
            cls.register_head(name, head_cls)
            return head_cls
        return decorator
