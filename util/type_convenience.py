

def list_broadcast(broadcast_to_list, resultant_list_length):
    if not isinstance(broadcast_to_list, list):
        broadcast_to_list = [broadcast_to_list]

    while len(broadcast_to_list) < resultant_list_length:
        broadcast_to_list += [ element for element in broadcast_to_list if len(broadcast_to_list) < resultant_list_length ]

    return broadcast_to_list
