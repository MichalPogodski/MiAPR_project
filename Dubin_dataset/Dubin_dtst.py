import dubins
import csv
import random


turning_radius = 1.0

with open('dubins_dataset.csv', mode='w') as dubins_dataset_file:
    dubins_dataset_writer = csv.writer(dubins_dataset_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    try:
        while True:
            x_start = 0
            y_start = 0
            theta_start = random.uniform(0, 360)
            x_finish = random.uniform(-1000, 1000)
            y_finish = random.uniform(-1000, 1000)
            theta_finish = random.uniform(0, 360)
            start_point = (x_start, y_start, theta_start)
            finish_point = (x_finish, y_finish, theta_finish)

            path = dubins.shortest_path(start_point, finish_point, turning_radius)
            distance = path.path_length()
            print(distance)
            #dubins_dataset_writer.writerow([x_start, y_start, theta_start, x_finish, y_finish, theta_finish, distance])
            dubins_dataset_writer.writerow([theta_start, x_finish, y_finish, theta_finish, distance])

    except KeyboardInterrupt:
        print('END')
