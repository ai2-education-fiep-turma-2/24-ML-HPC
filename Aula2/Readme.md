## instalação do spark

* instalando pyspark

pip install pyspark

* Instalando versão para cluster

    * download do arquivo de instalação https://spark.apache.org/downloads.html
    * descompactar em uma pasta local

* Seleciona uma máquina para ser master e uma ou mais para serem workers

* No nó master:

    * Selecione uma porta para servir de comunicação entre master e workers. Nesse exemplo a seguir 8893
    * defina o ip do nó master nesse caso 192.168.0.1
    * defina o ip do nó locaal nesse caso 192.168.0.1

Inclua essas variáveis no .bashrc so seu usuário /home/seuuser/.bashrc

```
export SPARK_MASTER_PORT="8893"
export SPARK_MASTER_HOST="192.168.0.1"
export SPARK_LOCAL_IP="192.168.0.1"
```

* Na pasta que descompactou inicie o master:

```
cd /home/silvio/sparksource/spark-3.1.1-bin-hadoop3.2/sbin
./start-master.sh
```

* Nos nós workers:

```
cd /home/silvio/sparksource/spark-3.1.1-bin-hadoop3.2/sbin
./start-worker.sh spark://192.168.0.1:8893
```

* Submetendo uma tarefa para o cluster spark:

```
/home/silvio/sparksource/spark-3.1.1-bin-hadoop3.2/bin/spark-submit --master spark://10.46.0.14:8893 Kmeans-spark.py
```
* Definindo parâmetros de quantidade de recursos e memória 

* Nesse caso a máquina precisa ter no mínimo 50 gb de ram para rodar a aplicação:

```
time /home/silvio/sparksource/spark-3.1.1-bin-hadoop3.2/bin/spark-submit --driver-memory 50g --num-executors 200 --executor-memory 50g --master spark://10.46.0.14:8893 /home/silvio/git/24-ML-HPC/Aula2/Spark-Copy11.py 
```

* Nesse caso a máquina precisa ter no mínimo 150 gb de ram para rodar a aplicação:

```
time /home/silvio/sparksource/spark-3.1.1-bin-hadoop3.2/bin/spark-submit --driver-memory 150g --num-executors 90 --executor-memory 150g --master spark://10.46.0.14:8893 /home/silvio/git/24-ML-HPC/Aula2/Spark-Copy11.py 
```

* É possível controlar a quantidade de cores que vai ser usada, mas é necessário instalar o Apache YARN (https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/YARN.html)

* Exemplo de uso: 
    * Predição de questões que serão encerradas no Stackoverflow (https://www.kaggle.com/c/predict-closed-questions-on-stack-overflow)

* Notebook do modelo (Preparação de Dados e Modelo): stackoverflowproject.ipynb

* Parquet

* Versão Para spark 

* Comparando desempenho entre Scikit-Learn e Spark (apenas preparação de dados para o problema StackOverflow
    * Spark-Copy11-PREP.py
    * stackoverflowproject-PREP.py
    
* Tempo de execução:

    * Preparação Apenas
        * 34 s - spark
        ```
        time /home/silvio/sparksource/spark-3.1.1-bin-hadoop3.2/bin/spark-submit --driver-memory 60g --executor-memory 60g --master spark://192.168.0.1:8893 Spark-prep.py
        ```
        
        * 84 s - Pandas + scikit-learn
        ```
        time python  stackoverflowproject-PREP.py
        ```
* Abrindo dataset preparado pelo spark no pandas:

```
dfparquet=pd.read_parquet('/home/silvio/stackOverflow-prep.parquet')
dfparquet.describe()
```
