# HPC

### Vetorização

* O3 opção para introduzir instruções vetoriais
* -S visualizar código assembly gerado

```
gcc -S -o prog.s stride-test.c
gcc -O3 -S -o progVEC.s stride-test.c
```

* comparando tempo de execução de programa vetorizado e não-vetorizado

```
gcc -O3 -o strid stride-test.c
gcc -O3 -o stridVEC stride-test.c
gcc -o strid stride-test.c
time ./strid
time ./stridVEC 
```


### Openmp

* exemplo 1
```
gcc OMP-hello.c -o OMP-helloOM -fopenmp
./OMP-helloOM
gcc OMP-hello.c -o OMP-hello
./OMP-hello
```

* exemplo 2

```
gcc OMP-matrix-sum.c -o OMP-matrix-sumOM
time ./OMP-matrix-sumOM
gcc OMP-matrix-sum.c -o OMP-matrix-sumOM -fopenmp
time ./OMP-matrix-sumOM 
```

### openACC

* usando compilador pgcc da nvidia para compilar código que delega parte das tarefas para GPU

```
pgcc vecadd.c -o vecaddACC -acc -Minfo=acc  -fast 
```

### Execução paralela do Tensorflow

