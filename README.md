# Testing the CAS GPU processing in Viya 4 with a simple SAS program

Reference: [Add a CAS "GPU-enabled" Node pool to boost your SAS Viya Analytics Platform !](https://communities.sas.com/t5/SAS-Communities-Library/Add-a-CAS-GPU-enabled-Node-pool-to-boost-your-SAS-Viya-Analytics/ta-p/809688)

The referenced blog, above, explains how to add and configure an extra CAS server with GPU processing in your Viya environment. 

Here are two ways to test and validate (from SAS Studio) that CAS can use the GPU processor (GAN and RNN with Word vector model). 

## Validate the GPU with a dummy program

* Open SAS Studio

* Login and run the following program.

```sas
options cashost='sas-cas-server-shared-design-gpu-client';
cas cas_gpu;

caslib _ALL_ assign;

data casuser.hmeq;
	set sampsio.hmeq;
run;

/* Define the runloop macro */
 %macro runloop;
     /* Specify all interval input variables*/
     %let names=value clage;
     /* Loop over all variables that need centroids generation */
     %do i=1 %to %sysfunc(countw(&names));
         %let name&i = %scan(&names, &i, %str( ));
         /* Call the GMM action to cluster each variable */
         proc cas ;
             action nonParametricBayes.gmm result=R/
                 table       = {name="hmeq"},
                 inputs      = {"&&name&i"},/*'value'*/
                 seed        = 1234567890,
                 maxClusters = 10,
                 alpha       = 1,
                 infer       = {method="VB",
                                maxVbIter =30,
                                covariance="diagonal",
                                threshold=0.01},
                 output      = {casOut={name='Score', replace=true},
                                copyVars={'value'}},
                 display     = {names={ "ClusterInfo"}}
                ;
             run;
             saveresult R.ClusterInfo replace dataset=work.weights&i;
         run;
         quit;

         /* Save variable name, weights, mean,     */
         /* and standard deviation of each cluster */
         data  weights&i;
             varname = "&&name&i";
             set  weights&i(rename=(&&name&i.._Mean=Mean
                                    &&name&i.._Variance=Var));
             /* Calculate standard deviation from variance*/
             std = sqrt(Var);
             drop Var;
         run;

         /* Construct centroids table from saved weights */
         %if &i=1 %then %do;
             data centroids;
             set weights&i;
             run;
         %end;
         %else %do;
             data centroids;
             set centroids weights&i;
             run;
         %end;
     %end;
 %mend;

 /* Run the runloop macro to generate the centroids table */
 %runloop;

data casuser.centroids;
   set centroids;
run;


proc cas;
     loadactionset "generativeAdversarialNet";
     action tabularGanTrain result = r /
         table           = {name = "hmeq",
                            vars = {'bad','value','clage','job'}},
         centroidsTable  = "centroids",
         nominals        = {"bad","job"},
         gpu             = {useGPU = True, device = 0},
         optimizerAe     = {method = "ADAM", numEpochs = 3},
         optimizerGan    = {method = "ADAM", numEpochs = 5},
         seed            = 12345,
         scoreSeed       = 0,
         numSamples      = 5,
         saveState       = {name = 'cpctStore', replace = True},
         casOut          = {name = 'out', replace = True};
     print r;
 run;
 quit;
```



## Apply a Word Vector Model to Score Documents and Train a Recurrent Neural Network (RNN) Model

* References:
  * [Apply a Word Vector Model to Score Documents and Train a Recurrent Neural Network (RNN) Model Using the tpWordVector Action] https://documentation.sas.com/doc/en/pgmsascdc/v_025/casvtapg/n047bll9q5h0nln1uulc1qpo0b2y.htm
  * [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)

* Copy the data in CAS pod with kubectl commands

  * First get them from the GitHub project or the stanford.edu page

    ```sh
    mkdir -p /tmp/GPU-HO-Datasets
    wget -O /tmp/GPU-HO-Datasets/reviews_test_100.csv https://raw.githubusercontent.com/raphaelpoumarede/testing-cas-gpu-processing/main/data/reviews_test_100.csv
    wget -O /tmp/GPU-HO-Datasets/reviews_train_5000.csv https://raw.githubusercontent.com/raphaelpoumarede/testing-cas-gpu-processing/main/data/reviews_train_5000.csv
    # Download the pre-trained word vectors dictionnary
    wget -O /tmp/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
    ```

  * Prepate the GLOVE dataset

    ```sh
    unzip /tmp/glove.6B.zip -d /tmp
    # insert a header line
    sed -i '1 i \term _1_ _2_ _3_ _4_ _5_ _6_ _7_ _8_ _9_ _10_ _11_ _12_ _13_ _14_ _15_ _16_ _17_ _18_ _19_ _20_ _21_ _22_ _23_ _24_ _25_ _26_ _27_ _28_ _29_ _30_ _31_     _32_ _33_ _34_ _35_ _36_ _37_ _38_ _39_ _40_ _41_ _42_ _43_ _44_ _45_ _46_ _47_ _48_ _49_ _50_ _51_ _52_ _53_ _54_ _55_ _56_ _57_ _58_ _59_ _60_ _61_ _62_ _63_         _64_ _65_ _66_ _67_ _68_ _69_ _70_ _71_ _72_ _73_ _74_ _75_ _76_ _77_ _78_ _79_ _80_ _81_ _82_ _83_ _84_ _85_ _86_ _87_ _88_ _89_ _90_ _91_ _92_ _93_ _94_ _95_         _96_ _97_ _98_ _99_ _100_' /tmp/glove.6B.100d.txt
    # remove the 10th row that starts with "", so CAS can load the file.
    sed -i '10d' /tmp/glove.6B.100d.txt
    
    # As an alternative you can download the clean dataset from this GitHub project
    # wget -O /tmp/glove_100d_tab_clean.txt.gz https://github.com/raphaelpoumarede/testing-cas-gpu-processing/releases/download/v1.0/glove_100d_tab_clean.txt.gz
    # gunzip -c /tmp/glove_100d_tab_clean.txt.gz > /tmp/glove_100d_tab_clean.txt
    ```


  * Then use the kubectl command to transfer the data in the Public CASlib through the CAS pod

    ```sh
    kubectl -n casgpu -c cas cp /tmp/GPU-HO-Datasets/reviews_test_100.csv sas-cas-server-shared-casgpu-controller:/cas/data/caslibs/public/reviews_test_100.csv
    kubectl -n casgpu -c cas cp /tmp/GPU-HO-Datasets/reviews_train_5000.csv sas-cas-server-shared-casgpu-controller:/cas/data/caslibs/public/reviews_train_5000.csv
    kubectl -n casgpu -c cas cp /tmp/glove.6B.100d.txt sas-cas-server-shared-casgpu-controller:/cas/data/caslibs/public/glove_100d_tab_clean.txt
    ```

* Open SAS Studio

* Login and run the following program.

```sas
/*Open a session on the CAS GPU server*/
cas MySession sessopts=(caslib=casuser timeout=1800 locale="en_US" metrics=true) host="sas-cas-server-shared-casgpu-client";
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

run;
quit;

proc cas;
   session MySession;
loadtable path='glove_100d_tab_clean.txt' caslib="PUBLIC"
importOptions={fileType='delimited' delimiter=' ' varChars=True}
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

   table.fetch /
      maxRows=20
      table={name="glove"};
   run;

table.columnInfo /
     table={name="glove"};
   run;

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

* If everything went well, you should see this kind of message in the SAS Log that confirms that the GPU processor is used by CAS:

```sh
Active Session now MySession.
NOTE: Executing action 'deepLearn.dlTrain'.
NOTE: Using controller.sas-cas-server-shared-casgpu.casgpu.svc.cluster.local: 1 out of 1 available GPU devices.
NOTE: Action 'deepLearn.dlTrain' used (Total process time):
NOTE:       real time               97.285795 seconds
NOTE:       cpu time                108.372646 seconds (111.40%)
NOTE:       total nodes             1 (8 cores)
NOTE:       total memory            51.00G
```
