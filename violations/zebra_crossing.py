ZEBRA_LINE_Y = 500

def check_zebra_violation(vehicle_y):

    if vehicle_y > ZEBRA_LINE_Y:
        return True

    return False