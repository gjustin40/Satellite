from utils import running_time
import time

a = running_time(10)

for i in range(10):
    start = time.time()
    time.sleep(1)
    i += 1
    end = time.time()
    interval_time, remain_time = a.predict(start, end, i)
    print(i, interval_time, remain_time)


import datetime

# create a timedelta object with microseconds
td = datetime.timedelta(seconds=10, microseconds=500000)
print(td)
print(type(td))
# create a new timedelta object with microseconds set to 0
td_without_microseconds = datetime.timedelta(seconds=td.seconds)

print(td)  # output: 0:00:10.500000
print(td_without_microseconds)  # output: 0:00:10
