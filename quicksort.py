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

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def measure_runtime(sort_function, n_values):
    runtimes = []
    for n in n_values:
        arr = [random.randint(0, 1000) for _ in range(n)]
        start_time = time.time()
        sort_function(arr)
        end_time = time.time()
        runtimes.append(end_time - start_time)
    return runtimes

n_values = list(range(100, 10100, 1000))
bubble_runtimes = measure_runtime(bubble_sort, n_values)
quicksort_runtimes = measure_runtime(quick_sort, n_values)

plt.plot(n_values, bubble_runtimes, marker='o', linestyle='-', label='Bubble Sort')
plt.plot(n_values, quicksort_runtimes, marker='s', linestyle='-', label='Quick Sort')
plt.xlabel('Array-Größe (n)')
plt.ylabel('Laufzeit (Sekunden)')
plt.title('Laufzeit von Bubble Sort vs. Quick Sort')
plt.legend()
plt.grid()
plt.show()

