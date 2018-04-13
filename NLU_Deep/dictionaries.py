indx2user_replies = { 0:["I want to travel from $DCITY$ to $ACITY$ on $DATE$ at $TIME$","I want to travel from $DCITY$ to $ACITY$ on $DATE$","I want to travel from $DCITY$ to $ACITY$","I want to travel to $ACITY$","flights from $DCITY$ to $ACITY$ on $DATE$ at $TIME$","flights from $DCITY$ to $ACITY$ on $DATE$","flights from $DCITY$ to $ACITY$"],
1:["I want to travel from $DCITY$","from $DCITY$","$DCITY$","flights from $DCITY$"],
2:["I want to travel to $ACITY$","to $ACITY$","to $ACITY$","flights to $ACITY$"],
4:["at $TIME$","around $TIME$","flights at about $TIME$"],
3:["I want to travel on $DATE$","on $DATE$","need flights for $DATE$"], # removed "need flights for $DATE$"
5:["I want to travel in $CLASS$ class","$CLASS$ class","$CLASS$"],
6:["I want to travel from $DCITY$ to $ACITY$","flights from $DCITY$ to $ACITY$"],
7:["Need to travel on $DATE$ at $TIME$","On $DATE$ at $TIME$","flights on $DATE$ at $TIME$"],
8:["Yes","Yes please"],
9:["Yes","Yes please"],
10:["Yes","Yes please"],
11:["Yes","Yes please"],
12:["Yes","Yes please"],
13:["Thanks"]

}

tags2values = {"$DCITY$":["boston","pittsburgh","washington","tacoma"],
"$ACITY$":["denver","baltimore","orlando","philadelphia"],
"$TIME$":["evening","morning","afternoon"],
"$DATE$":["twenty eight december","fourth january"], # removed ,"fifteenth june"
"$CLASS$":["business","economy"]
}



actions_sentences = ["Hello How may I help you?",  # action : 0 greet
                     "Please tell the city of Departure?",  # action : 1 ask dept city
                     "Please tell the city of Arrival?",  # action : 2 ask arrival city
                     "Please tell the date of departure?",  # action : 3 ask departure day
                     "Please tell the time of departure?",  # action : 4 ask dep time
                     "Please specify the class of flight?",  # action : 5 ask class of travel
                     "From where to where?",  # action : 6 ask Departure and Arrival City
                     "Specify the time and date?",  # action : 7 ask date and time both
                     "Are travelling from $DCITY$?",  # action : 8 reask dept city
                     "Are you travelling to $ACITY$?",  # action : 9 reask arrival city
                     "Are you travelling on $DATE$?",  # action : 10 reask dept day
                     "Are you travelling at $TIME$?",  # action : 11 reask dept time
                     "You would like to travel via $CLASS$?",  # action : 12 reask class
                     "Here is you itenary\nYou are travelling from $DCITY$ to $ACITY$ on $DATE$ at $TIME$ via $CLASS$.\nThanks for using the flight attendant",  # action : 13 close the conversation
					 ]

tags = ['I-depart_date.day_name',
        'I-depart_date.day_number',  # (yes) Date
        'I-depart_date.month_name',  # (yes) Date
        # 'I-depart_date.year',# (yes) Date

        'I-depart_time.time',  # (yes)

        'I-fromloc.city_name',  # (yes)
        'I-fromloc.state_name',  # (yes) to city


        'I-toloc.city_name',  # (yes)
        'I-toloc.state_name',  # (yes) to ci]
        'I-class_type'
        ]
convert = {'I-depart_date.day_name': 'I-depart_date',
           'I-depart_date.day_number': 'I-depart_date',  # (yes) Date
           'I-depart_date.month_name': 'I-depart_date',  # (yes) Date
           # 'I-depart_date.year',# (yes) Date

           'I-depart_time.time': 'I-depart_time.time',  # (yes)

           'I-fromloc.city_name': 'I-fromloc.city_name',  # (yes)
           'I-fromloc.state_name': 'I-fromloc.city_name',  # (yes) to city


           'I-toloc.city_name': 'I-toloc.city_name',  # (yes)
           'I-toloc.state_name': 'I-toloc.city_name',  # (yes) to ci]
           'I-class_type': 'I-class_type',
           'O': 'O'
           }
# slots = ['$ACITY$', '$DCITY$','$DATE$', '$TIME$','$CLASS$' ]

atis_labels = { 0: 'B-aircraft_code',
 1: 'B-airline_code',
 2: 'B-airline_name',
 3: 'B-airport_code',
 4: 'B-airport_name',
 5: 'B-arrive_date.date_relative',
 6: 'B-arrive_date.day_name',
 7: 'B-arrive_date.day_number',
 8: 'B-arrive_date.month_name',
 9: 'B-arrive_date.today_relative',
 10: 'B-arrive_time.end_time',
 11: 'B-arrive_time.period_mod',
 12: 'B-arrive_time.period_of_day',
 13: 'B-arrive_time.start_time',
 14: 'B-arrive_time.time',
 15: 'B-arrive_time.time_relative',
 16: 'B-booking_class',
 17: 'B-city_name',
 18: 'B-class_type',
 19: 'B-compartment',
 20: 'B-connect',
 21: 'B-cost_relative',
 22: 'B-day_name',
 23: 'B-day_number',
 24: 'B-days_code',
 25: 'B-depart_date.date_relative',
 26: 'B-depart_date.day_name',
 27: 'B-depart_date.day_number',
 28: 'B-depart_date.month_name',
 29: 'B-depart_date.today_relative',
 30: 'B-depart_date.year',
 31: 'B-depart_time.end_time',
 32: 'B-depart_time.period_mod',
 33: 'B-depart_time.period_of_day',
 34: 'B-depart_time.start_time',
 35: 'B-depart_time.time',
 36: 'B-depart_time.time_relative',
 37: 'B-economy',
 38: 'B-fare_amount',
 39: 'B-fare_basis_code',
 40: 'B-flight',
 41: 'B-flight_days',
 42: 'B-flight_mod',
 43: 'B-flight_number',
 44: 'B-flight_stop',
 45: 'B-flight_time',
 46: 'B-fromloc.airport_code',
 47: 'B-fromloc.airport_name',
 48: 'B-fromloc.city_name',
 49: 'B-fromloc.state_code',
 50: 'B-fromloc.state_name',
 51: 'B-meal',
 52: 'B-meal_code',
 53: 'B-meal_description',
 54: 'B-mod',
 55: 'B-month_name',
 56: 'B-or',
 57: 'B-period_of_day',
 58: 'B-restriction_code',
 59: 'B-return_date.date_relative',
 60: 'B-return_date.day_name',
 61: 'B-return_date.day_number',
 62: 'B-return_date.month_name',
 63: 'B-return_date.today_relative',
 64: 'B-return_time.period_mod',
 65: 'B-return_time.period_of_day',
 66: 'B-round_trip',
 67: 'B-state_code',
 68: 'B-state_name',
 69: 'B-stoploc.airport_code',
 70: 'B-stoploc.airport_name',
 71: 'B-stoploc.city_name',
 72: 'B-stoploc.state_code',
 73: 'B-time',
 74: 'B-time_relative',
 75: 'B-today_relative',
 76: 'B-toloc.airport_code',
 77: 'B-toloc.airport_name',
 78: 'B-toloc.city_name',
 79: 'B-toloc.country_name',
 80: 'B-toloc.state_code',
 81: 'B-toloc.state_name',
 82: 'B-transport_type',
 83: 'I-airline_name',
 84: 'I-airport_name',
 85: 'I-arrive_date.day_number',
 86: 'I-arrive_time.end_time',
 87: 'I-arrive_time.period_of_day',
 88: 'I-arrive_time.start_time',
 89: 'I-arrive_time.time',
 90: 'I-arrive_time.time_relative',
 91: 'I-city_name',
 92: 'I-class_type',
 93: 'I-cost_relative',
 94: 'I-depart_date.day_number',
 95: 'I-depart_date.today_relative',
 96: 'I-depart_time.end_time',
 97: 'I-depart_time.period_of_day',
 98: 'I-depart_time.start_time',
 99: 'I-depart_time.time',
 100: 'I-depart_time.time_relative',
 101: 'I-economy',
 102: 'I-fare_amount',
 103: 'I-fare_basis_code',
 104: 'I-flight_mod',
 105: 'I-flight_number',
 106: 'I-flight_stop',
 107: 'I-flight_time',
 108: 'I-fromloc.airport_name',
 109: 'I-fromloc.city_name',
 110: 'I-fromloc.state_name',
 111: 'I-meal_code',
 112: 'I-meal_description',
 113: 'I-restriction_code',
 114: 'I-return_date.date_relative',
 115: 'I-return_date.day_number',
 116: 'I-return_date.today_relative',
 117: 'I-round_trip',
 118: 'I-state_name',
 119: 'I-stoploc.city_name',
 120: 'I-time',
 121: 'I-today_relative',
 122: 'I-toloc.airport_name',
 123: 'I-toloc.city_name',
 124: 'I-toloc.state_name',
 125: 'I-transport_type',
 126: 'O'}

slots = ['$ACITY$', '$DCITY$','$DATE$', '$TIME$','$CLASS$','$NULL$' ]

labels2labels={'B-aircraft_code':slots[5],
 'B-airline_code':slots[5],
 'B-airline_name':slots[5],
 'B-airport_code':slots[5],
 'B-airport_name':slots[5],
 'B-arrive_date.date_relative':slots[5],
 'B-arrive_date.day_name':slots[5],
 'B-arrive_date.day_number':slots[5],
 'B-arrive_date.month_name':slots[5],
 'B-arrive_date.today_relative':slots[5],
 'B-arrive_time.end_time':slots[5],
 'B-arrive_time.period_mod':slots[5],
 'B-arrive_time.period_of_day':slots[5],
 'B-arrive_time.start_time':slots[5],
 'B-arrive_time.time':slots[5],
 'B-arrive_time.time_relative':slots[5],
 'B-booking_class':slots[4],
 'B-city_name':slots[5],
 'B-class_type':slots[4],
 'B-compartment':slots[5],
 'B-connect':slots[5],
 'B-cost_relative':slots[5],
 'B-day_name':slots[5],
 'B-day_number':slots[5],
 'B-days_code':slots[5],
 'B-depart_date.date_relative':slots[5], # converted from 2 -> 5
 'B-depart_date.day_name':slots[2],
 'B-depart_date.day_number':slots[2],
 'B-depart_date.month_name':slots[2],
 'B-depart_date.today_relative':slots[5], # converted from 2 -> 5
 'B-depart_date.year':slots[2],
 'B-depart_time.end_time':slots[3],
 'B-depart_time.period_mod':slots[3],
 'B-depart_time.period_of_day':slots[3],
 'B-depart_time.start_time':slots[3],
 'B-depart_time.time':slots[3],
 'B-depart_time.time_relative':slots[5], # converted from 3 -> 5
 'B-economy':slots[4],
 'B-fare_amount':slots[5],
 'B-fare_basis_code':slots[5],
 'B-flight':slots[5],
 'B-flight_days':slots[5],
 'B-flight_mod':slots[5],
 'B-flight_number':slots[5],
 'B-flight_stop':slots[5],
 'B-flight_time':slots[5],
 'B-fromloc.airport_code':slots[5],
 'B-fromloc.airport_name':slots[5],
 'B-fromloc.city_name':slots[1],
 'B-fromloc.state_code':slots[1],
 'B-fromloc.state_name':slots[1],
 'B-meal':slots[5],
 'B-meal_code':slots[5],
 'B-meal_description':slots[5],
 'B-mod':slots[5],
 'B-month_name':slots[5],
 'B-or':slots[5],
 'B-period_of_day':slots[5],
 'B-restriction_code':slots[5],
 'B-return_date.date_relative':slots[5],
 'B-return_date.day_name':slots[5],
 'B-return_date.day_number':slots[5],
 'B-return_date.month_name':slots[5],
 'B-return_date.today_relative':slots[5],
 'B-return_time.period_mod':slots[5],
 'B-return_time.period_of_day':slots[5],
 'B-round_trip':slots[5],
 'B-state_code':slots[5],
 'B-state_name':slots[5],
 'B-stoploc.airport_code':slots[5],
 'B-stoploc.airport_name':slots[5],
 'B-stoploc.city_name':slots[5],
 'B-stoploc.state_code':slots[5],
 'B-time':slots[5],
 'B-time_relative':slots[5],
 'B-today_relative':slots[5],
 'B-toloc.airport_code':slots[5],
 'B-toloc.airport_name':slots[5],
 'B-toloc.city_name':slots[0],
 'B-toloc.country_name':slots[0],
 'B-toloc.state_code':slots[0],
 'B-toloc.state_name':slots[0],
 'B-transport_type':slots[5],
 'I-airline_name':slots[5],
 'I-airport_name':slots[5],
 'I-arrive_date.day_number':slots[5],
 'I-arrive_time.end_time':slots[5],
 'I-arrive_time.period_of_day':slots[5],
 'I-arrive_time.start_time':slots[5],
 'I-arrive_time.time':slots[5],
 'I-arrive_time.time_relative':slots[5],
 'I-city_name':slots[5],
 'I-class_type':slots[4],
 'I-cost_relative':slots[5],
 'I-depart_date.day_number':slots[2],
 'I-depart_date.today_relative':slots[2],
 'I-depart_time.end_time':slots[3],
 'I-depart_time.period_of_day':slots[3],
 'I-depart_time.start_time':slots[3],
 'I-depart_time.time':slots[3],
 'I-depart_time.time_relative':slots[3],
 'I-economy':slots[4],
 'I-fare_amount':slots[5],
 'I-fare_basis_code':slots[5],
 'I-flight_mod':slots[5],
 'I-flight_number':slots[5],
 'I-flight_stop':slots[5],
 'I-flight_time':slots[5],
 'I-fromloc.airport_name':slots[5],
 'I-fromloc.city_name':slots[1],
 'I-fromloc.state_name':slots[1],
 'I-meal_code':slots[5],
 'I-meal_description':slots[5],
 'I-restriction_code':slots[5],
 'I-return_date.date_relative':slots[5],
 'I-return_date.day_number':slots[5],
 'I-return_date.today_relative':slots[5],
 'I-round_trip':slots[5],
 'I-state_name':slots[5],
 'I-stoploc.city_name':slots[5],
 'I-time':slots[5],
 'I-today_relative':slots[5],
 'I-toloc.airport_name':slots[5],
 'I-toloc.city_name':slots[0],
 'I-toloc.state_name':slots[0],
 'I-transport_type':slots[5],
'O':slots[5]
}
