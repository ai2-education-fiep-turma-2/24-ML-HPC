# 24-ML-HPC

## Hadoop

* instalação

```
mkdir hadoop
cd $HOME
wget https://downloads.apache.org/hadoop/common/hadoop-3.2.2/hadoop-3.2.2.tar.gz
tar -xvzf hadoop-3.2.2.tar.gz 
```

* configuração do ambiente JAVA e PATH
   * Caso não tenha o JAVA_HOME declarado
   
```
echo $JAVA_HOME
which java
export JAVA_HOME=/usr/bin/
export PATH=$HOME/hadoop/hadoop-3.2.2/bin:$PATH
```

* Rodando aplicação World Count

```
mkdir hadoop/jobs/input
```
* copie os arquivos 1.txt, 2.txt e 3.txt para $HOME/hadoop/jobs/input
* copie o arquivo WordCount.java para $HOME/hadoop/jobs/

```
hadoop com.sun.tools.javac.Main WordCount.java
jar cf wc.jar WordCount*.class
hadoop jar wc.jar WordCount /home/silvio/hadoop/jobs/input /home/silvio/hadoop/jobs/output
```

* Um arquivo de saída será gerado na pasta output 

## Tensorflow Distribuído


