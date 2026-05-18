#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// Print Array
void printArray(vector<int> arr) {
    for(int x : arr) {
        cout << x << " ";
    }
    cout << endl;
}

// Sequential Bubble Sort
void bubbleSort(vector<int>& arr) {
    int n = arr.size();

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n - i - 1; j++) {
            if(arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Parallel Bubble Sort (Odd-Even Sort)
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();

    for(int i = 0; i < n; i++) {

        // Even Phase
        #pragma omp parallel for
        for(int j = 0; j < n - 1; j += 2) {
            if(arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }

        // Odd Phase
        #pragma omp parallel for
        for(int j = 1; j < n - 1; j += 2) {
            if(arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Merge Function
void merge(vector<int>& arr, int l, int m, int r) {
    vector<int> temp;

    int i = l;
    int j = m + 1;

    while(i <= m && j <= r) {
        if(arr[i] < arr[j]) {
            temp.push_back(arr[i++]);
        }
        else {
            temp.push_back(arr[j++]);
        }
    }

    while(i <= m) {
        temp.push_back(arr[i++]);
    }

    while(j <= r) {
        temp.push_back(arr[j++]);
    }

    for(int k = l; k <= r; k++) {
        arr[k] = temp[k - l];
    }
}

// Sequential Merge Sort
void mergeSort(vector<int>& arr, int l, int r) {
    if(l < r) {
        int m = (l + r) / 2;

        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        merge(arr, l, m, r);
    }
}

// Parallel Merge Sort
void parallelMergeSort(vector<int>& arr, int l, int r) {
    if(l < r) {
        int m = (l + r) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, l, m);

            #pragma omp section
            parallelMergeSort(arr, m + 1, r);
        }

        merge(arr, l, m, r);
    }
}

int main() {

    vector<int> data = {8, 4, 6, 2, 9, 1, 5};

    cout << "Original Array: ";
    printArray(data);

    double start, end;

    // Bubble Sort
    vector<int> a = data;
    vector<int> b = data;

    start = omp_get_wtime();
    bubbleSort(a);
    end = omp_get_wtime();

    cout << "\nSequential Bubble Sort: ";
    printArray(a);
    cout << "Execution Time: " << end - start << " sec" << endl;

    start = omp_get_wtime();
    parallelBubbleSort(b);
    end = omp_get_wtime();

    cout << "\nParallel Bubble Sort: ";
    printArray(b);
    cout << "Execution Time: " << end - start << " sec" << endl;

    // Merge Sort
    vector<int> c = data;
    vector<int> d = data;

    start = omp_get_wtime();
    mergeSort(c, 0, c.size() - 1);
    end = omp_get_wtime();

    cout << "\nSequential Merge Sort: ";
    printArray(c);
    cout << "Execution Time: " << end - start << " sec" << endl;

    start = omp_get_wtime();
    parallelMergeSort(d, 0, d.size() - 1);
    end = omp_get_wtime();

    cout << "\nParallel Merge Sort: ";
    printArray(d);
    cout << "Execution Time: " << end - start << " sec" << endl;

    return 0;
}
