SLOTS = ['05:00-07:00','07:00-09:00','09:00-11:00','11:00-13:00','13:00-15:00','15:00-17:00','17:00-19:00','19:00-21:00','21:00-23:00','23:00-01:00','01:00-03:00','03:00-05:00']


def time_slot_for_hour(hour):
    hour = int(hour) % 24
    start = ((hour - 5) // 2 * 2 + 5) % 24
    end = (start + 2) % 24
    return f'{start:02d}:00-{end:02d}:00'


def hour_for_slot(slot):
    return int(slot[:2])
