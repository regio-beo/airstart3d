import time
import random
import matplotlib.pyplot as plt

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        if not swapped:
            break

def measure_runtime(n_values):
    runtimes = []
    for n in n_values:
        arr = [random.randint(0, 1000) for _ in range(n)]
        start_time = time.time()
        bubble_sort(arr)
        end_time = time.time()
        runtimes.append(end_time - start_time)
    return runtimes

n_values = list(range(100, 1100, 100))
runtimes = measure_runtime(n_values)

plt.plot(n_values, runtimes, marker='o', linestyle='-')
plt.xlabel('Array-Größe (n)')
plt.ylabel('Laufzeit (Sekunden)')
plt.title('Laufzeit von Bubble Sort')
plt.grid()
plt.show()
