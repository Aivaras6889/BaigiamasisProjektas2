CLASS_NAMES = {
    0: 'Speed Limit 20',
    1: 'Speed Limit 30',
    2: 'Speed Limit 50',
    3: 'Speed Limit 60',
    4: 'Speed Limit 70',
    5: 'Speed Limit 80',
    6: 'End Speed Limit 80',
    7: 'Speed Limit 100',
    8: 'Speed Limit 120',
    9: 'No Passing',
    10: 'No Passing for Vehicles Over 3.5 Metric Tons',
    11: 'Right-of-Way at Next Intersection',
    12: 'Priority Road',
    13: 'Yield',
    14: 'Stop',
    15: 'No Vehicles',
    16: 'Vehicles Over 3.5 Metric Tons Prohibited',
    17: 'No Entry',
    18: 'General Caution',
    19: 'Dangerous Curve Left',
    20: 'Dangerous Curve Right',
    21: 'Double Curve',
    22: 'Bumpy Road',
    23: 'Slippery Road',
    24: 'Road Narrows on the Right',
    25: 'Road Work',
    26: 'Traffic Signals',
    27: 'Pedestrians',
    28: 'Children Crossing',
    29: 'Bicycles Crossing',
    30: 'Beware of Ice/Snow',
    31: 'Wild Animals Crossing',
    32: 'End Speed + Passing Limits',
    33: 'Turn Right Ahead',
    34: 'Turn Left Ahead',
    35: 'Ahead Only',
    36: 'Go Straight or Right',
    37: 'Go Straight or Left',
    38: 'Keep Right',
    39: 'Keep Left',
    40: 'Roundabout Mandatory',
    41: 'End of No Passing',
    42: 'End No Passing Veh > 3.5 Tons'
}

def get_class_name(class_id):
    """Convert class ID to readable name"""
    return CLASS_NAMES.get(class_id, f'Unknown Class {class_id}')

def get_all_class_names():
    """Get all class names for dropdown/selection"""
    return CLASS_NAMES