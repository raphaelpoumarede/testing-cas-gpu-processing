/* Sample Program to test the GPU processor(s) without having to load any data*/
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
