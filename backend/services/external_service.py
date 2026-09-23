import os
import random


def weather_for(place, month, slot, frame):
    rows = frame[(frame.Location == place) & (frame.Month == month) & (frame.TimeSlot == slot)]
    if rows.empty:
        rows = frame[frame.Location == place]
    row = rows.iloc[0] if not rows.empty else frame.iloc[0]
    return {'weather_condition': row.Weather, 'temperature': float(row.Temperature), 'source': 'historical'}


def traffic_for(place, day, slot, frame):
    rows = frame[(frame.Location == place) & (frame.Day == day) & (frame.TimeSlot == slot)]
    if rows.empty:
        rows = frame[frame.Location == place]
    row = rows.iloc[0] if not rows.empty else frame.iloc[0]
    return {'traffic_level': row.Traffic, 'source': 'historical'}
