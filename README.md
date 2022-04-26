# testing-cas-gpu-processing

* Copy the data in CAS pod with kubectl commands

* First get them from the GitHub project or the stanford.edu page

    ```sh
    mkdir -p /tmp/GPU-HO-Datasets
    wget -O /tmp/GPU-HO-Datasets/reviews_test_100.csv https://raw.githubusercontent.com/raphaelpoumarede/testing-cas-gpu-processing/main/data/reviews_test_100.csv
    wget -O /tmp/GPU-HO-Datasets/reviews_train_5000.csv https://raw.githubusercontent.com/raphaelpoumarede/testing-cas-gpu-processing/main/data/reviews_train_5000.csv
    # Download the pre-trained word vectors dictionnary
    # wget -O /tmp/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
    # Instead we download the clean dataset
    wget -O /tmp/glove_100d_tab_clean.txt.gz https://github.com/raphaelpoumarede/testing-cas-gpu-processing/releases/download/v1.0/glove_100d_tab_clean.txt.gz
    gunzip -c /tmp/glove_100d_tab_clean.txt.gz > /tmp/glove_100d_tab_clean.txt
    ```

* Then use the kubectl command to transfer the data in the Public CASlib through the CAS pod

    ```sh
    kubectl -n casgpu -c cas cp /tmp/GPU-HO-Datasets/reviews_test_100.csv sas-cas-server-shared-casgpu-controller:/cas/data/caslibs/public/reviews_test_100.csv
    kubectl -n casgpu -c cas cp /tmp/GPU-HO-Datasets/reviews_train_5000.csv sas-cas-server-shared-casgpu-controller:/cas/data/caslibs/public/reviews_train_5000.csv
    kubectl -n casgpu -c cas cp /tmp/glove_100d_tab_clean.txt sas-cas-server-shared-casgpu-controller:/cas/data/caslibs/public/glove_100d_tab_clean.txt
    ```

* Open SAS Studio

* Login and run the following program.

```sas
/*Open a session on the CAS GPU server*/
cas MySession sessopts=(caslib=casuser timeout=1800 locale="en_US") host="sas-cas-server-shared-casgpu-client";
libname CASUSER cas caslib="CASUSER";

/************************/
/* LOAD THE DATA IN CAS */
/************************/
proc cas;
   session MySession;
loadtable path='reviews_train_5000.csv' caslib="PUBLIC"
casOut={name='reviews_train' replace=true}
importoptions={fileType='csv' varChars=True getNames=True};

loadtable path='reviews_test_100.csv' caslib="PUBLIC"
casOut={name='reviews_test' replace=true}
importoptions={fileType='csv' varChars=True getNames=True};

loadtable path='glove_100d_tab_clean.txt' caslib="PUBLIC"
importOptions={fileType='delimited' delimiter='\t' guessRows=2 varChars=True}
casOut={name='glove' replace=true};
run;
quit;

/**********************/
/* CHECK OUT THE DATA */
/**********************/
proc cas;
   session MySession;

  table.tableInfo /
    name="reviews_train";

  table.tableDetails /
    name="reviews_train";

  table.fetch /
      maxRows=20
      table={name="reviews_train"};

table.tableInfo /
     name="reviews_test";
   run;

   table.tableDetails /
     name="reviews_test";
   run;

   table.fetch /
      maxRows=10
      table={name="reviews_test"};
   run;

/*   table.fetch /
      maxRows=20
      table={name="glove"};
   run;*/
quit;

/***********************/
/* BUILD THE RNN MODEL */
/***********************/
proc cas;
  session MySession;

  deepLearn.buildmodel /
  modelTable={name="TestRNN", replace=TRUE}
             type="RNN";

  deepLearn.addLayer /
      layer={type="INPUT"}
      modelTable={name="TestRNN"}
      name="data";

  deepLearn.addLayer /
      layer={type="RECURRENT"
             n=50
             init="normal"
             rnnType="GRU"
             }
      modelTable={name="TestRNN"}
      name="rnn1"
      srcLayers={"data"};

  deepLearn.addLayer /
      layer={type="output"
             act="SOFTMAX"
             init="xavier"}
      modelTable={name="TestRNN"}
      name="outlayer"
      srcLayers={"rnn1"};


/* Take a peek at the model info */

  deepLearn.modelInfo /
      modelTable={name="TestRNN"};

  table.fetch /
      table={name="TestRNN"};

run;
quit;

/*******************/
/* TRAIN THE MODEL */
/*******************/

proc cas;
   session MySession;

   deepLearn.dlTrain /
      inputs={{name="review"}}
      modelTable={name="TestRNN"}
      modelWeights={name="reviewsTrainedWeights",
                    replace=TRUE
                   }
      nThreads=1
      /*Enable GPU processing*/
      GPU=TRUE
                            optimizer={algorithm={method="ADAM",
                            lrPolicy='step',
                            gamma=0.5,
                            beta1=0.9,
                            beta2=0.999,
                            learningRate=0.001
                           },
                 maxEpochs=10,
                 miniBatchSize=1
                }
      seed=12345
      table={name="reviews_train"}
      texts={{name="review"}}
      target="positive"
      textParms={initInputEmbeddings={importOptions={fileType="AUTO"},
                                      name="glove"
                                     }
                }
      ;
   run;
quit;


/*******************/
/* SCORE THE MODEL */
/*******************/

proc cas;
   session MySession;

   deepLearn.dlScore /
    /*Enable GPU processing*/
      GPU=TRUE
      /*gpu=dict(devices=0)*/
      casOut={name="train_scored",
              replace=TRUE
             }
      initWeights={name="reviewsTrainedWeights"}
      modelTable={name="TestRNN"}
      table={name="reviews_test"}
      textParms={initInputEmbeddings={importOptions={fileType="AUTO"},
                                      name="glove"
                                     }
                };
   run;
quit;

cas MySession terminate;
```
