from Attackers.SMARTAttacker import SmartAttacker

def loadAttacker(args):
    attacker = ''
    if args["attacker"] == 'SMART':
        attacker = SmartAttacker(args)
    else:
        print('No classifier created')

    return attacker