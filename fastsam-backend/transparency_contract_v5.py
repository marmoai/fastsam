"""Small dependency-free validation of whole-surface transmission intent."""
def transmission_request(contract):
    return (contract.get('schemaVersion')==1
            and contract.get('mode')=='recover_transmission'
            and contract.get('scope')=='whole'
            and contract.get('representation')=='physical'
            and contract.get('backgroundEvidence') in {
                'visible_through_surface','diffuse_transmission','uniform_backdrop_through_surface'}
            and isinstance(contract.get('confidence'),(int,float))
            and not isinstance(contract.get('confidence'),bool)
            and .8<=contract['confidence']<=1)
