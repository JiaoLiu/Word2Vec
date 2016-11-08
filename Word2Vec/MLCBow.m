//
//  MLCBow.m
//  Word2Vec
//
//  Created by Jiao Liu on 11/3/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import "MLCBow.h"
#import <Accelerate/Accelerate.h>

@implementation MLCBow

- (id)initWithLayerSize:(int)size iterNum:(int)num descentRate:(double)rate
{
    self = [super init];
    if (self) {
        _layerSize = size;
        _iterNum = num;
        _descentRate = rate;
        _isHS = true;
        _threadNum = 20;
        _windowSize = 5;
        _negativeSeed = 5;
    }
    return self;
}

- (id)init
{
    self = [super init];
    if (self) {
        _layerSize = 50;
        _iterNum = 5;
        _descentRate = 0.05;
        _isHS = true;
        _threadNum = 20;
        _windowSize = 5;
        _negativeSeed = 5;
    }
    return self;
}

#pragma mark - Main Method

- (void)startTrain:(NSString *)filePath
{
    _loader = [[MLFileLoader alloc] initWithFile:filePath];
    [_loader readWordFromFile];
    int size = _loader.dicSize * _layerSize;
    _startDescentRate = _descentRate;
    _trainedNum = 0;
    _descentGap = 1e4;
    _neur1 = calloc(size, sizeof(double));
    _neur2 = calloc(size, sizeof(double));
    for (int i = 0; i < size; i++) {
        _neur1[i] = (rand() / (double)RAND_MAX - 0.5) / _layerSize;
    }
    
    if (_loader.totalWords / _threadNum > MAX_SENTENCE) {
        _threadNum = _loader.totalWords % MAX_SENTENCE ? _loader.totalWords / MAX_SENTENCE + 1 : _loader.totalWords / MAX_SENTENCE;
    }
    
    if (!_isHS) {
        [self createNSTable];
    }
    
    dispatch_queue_t queue = dispatch_queue_create("CBOW", DISPATCH_QUEUE_SERIAL);
    dispatch_group_t group = dispatch_group_create();
    __block bool isDone = false;
    for (int i = 0; i < _threadNum; i++) {
        dispatch_group_async(group, queue, ^{
            if (_isHS) {
                [self HSCbow:i file:[filePath UTF8String]];
            }
            else
            {
                [self NSCbow:i file:[filePath UTF8String]];
            }
        });
    }
    dispatch_group_notify(group, queue, ^{
        isDone = true;
    });
    while (!isDone) {}
    
//    for (int a = 0; a < _loader.dicSize; a++) {
//        printf("%s ", _loader.dictionary[a].word);
//        for (int b = 0; b < _layerSize; b++) printf("%lf ", _neur1[a * _layerSize + b]);
//        printf("\n");
//    }
}

- (void)saveToFile:(NSString *)filePath
{
    FILE *file = fopen([filePath UTF8String], "w");
    for (int a = 0; a < _loader.dicSize; a++) {
        fprintf(file, "%s ", _loader.dictionary[a].word);
        for (int b = 0; b < _layerSize; b++) fprintf(file, "%lf ", _neur1[a * _layerSize + b]);
        fprintf(file,"\n");
    }
    fclose(file);
}

// Hierarchical Softmax
- (void)HSCbow:(int)threadId file:(const char *)filePath
{
    FILE *file = fopen(filePath, "rb");
    if (file == NULL) {
        NSLog(@"File Error");
        return;
    }
    fseek(file, _loader.fileSize / _threadNum * threadId, SEEK_SET);
    int sentence[MAX_SENTENCE];
    int senPos = 0;
    int iter = _iterNum;
    while (iter) {
        
        while (1) {
            int index = [_loader readWordIndex:file];
            if (feof(file)) {
                break;
            }
            if (index == -1) {
                continue;
            }
            sentence[senPos] = index;
            senPos++;
            _trainedNum++;
            if (senPos >= _loader.totalWords / _threadNum) {
                break;
            }
        }
        
        if (_trainedNum > _descentGap) {
            _descentRate *= 1 - _trainedNum / (double)(_iterNum * _loader.totalWords + 1);
            if (_descentRate < _startDescentRate / 1e4) {
                _descentRate = _startDescentRate / 1e4;
            }
            _descentGap += 1e4;
        }
        
        for (int i = 0; i < senPos; i++) {
            int word = sentence[i];
            int start = rand() %  _windowSize;
            double *layer = calloc(_layerSize, sizeof(double));
            double *layerLoss = calloc(_layerSize, sizeof(double));
            // sum the surrounding words
            for (int j = start; j < 2 * _windowSize + 1 - start; j ++) if(j != _windowSize){
                int cindex = i - _windowSize + j;
                if (cindex < 0 || cindex >= senPos) {
                    continue;
                }
                int cword = sentence[cindex];
                vDSP_vaddD(layer, 1, (_neur1 + _layerSize * cword), 1, layer, 1, _layerSize);
            }
            
            // update neur2
            for (int l = 0; l < _loader.dictionary[word].codeLength; l++) {
                int dw = _loader.dictionary[word].code[l];
                int codeP = _loader.dictionary[word].position[l];
                double multX = 0;
                for (int p = 0; p < _layerSize; p++) {
                    multX += layer[p] * _neur2[codeP*_layerSize + p];
                }
                double q = [self sigmoid:multX];
                double g = _descentRate * (1 - dw - q);
                for (int p = 0; p < _layerSize; p++) {
                    layerLoss[p] += g * _neur2[codeP*_layerSize + p];
                }
                
                for (int p = 0; p < _layerSize; p++) {
                    _neur2[codeP*_layerSize + p] += g * layer[p];
                }
            }
            
            // update neur1
            for (int j = start; j < 2 * _windowSize + 1 - start; j ++) if(j != _windowSize){
                int cindex = i - _windowSize + j;
                if (cindex < 0 || cindex >= senPos) {
                    continue;
                }
                int cword = sentence[cindex];
                vDSP_vaddD(layerLoss, 1, (_neur1 + _layerSize * cword), 1, (_neur1 + _layerSize * cword), 1, _layerSize);
            }
            
            free(layer);
            free(layerLoss);
        }
        
        senPos = 0;
        fseek(file, _loader.fileSize / _threadNum * threadId, SEEK_SET);
        iter--;
    }
}

// Negative Sampling
- (void)NSCbow:(int)threadId file:(const char *)filePath
{
    FILE *file = fopen(filePath, "rb");
    if (file == NULL) {
        NSLog(@"File Error");
        return;
    }
    fseek(file, _loader.fileSize / _threadNum * threadId, SEEK_SET);
    int sentence[MAX_SENTENCE];
    int senPos = 0;
    int iter = _iterNum;
    while (iter) {
        
        while (1) {
            int index = [_loader readWordIndex:file];
            if (feof(file)) {
                break;
            }
            if (index == -1) {
                continue;
            }
            sentence[senPos] = index;
            senPos++;
            _trainedNum++;
            if (senPos >= _loader.totalWords / _threadNum) {
                break;
            }
        }
        
        if (_trainedNum > _descentGap) {
            _descentRate *= 1 - _trainedNum / (double)(_iterNum * _loader.totalWords + 1);
            if (_descentRate < _startDescentRate / 1e4) {
                _descentRate = _startDescentRate / 1e4;
            }
            _descentGap += 1e4;
        }
        
        for (int i = 0; i < senPos; i++) {
            int word = sentence[i];
            int start = rand() %  _windowSize;
            double *layer = calloc(_layerSize, sizeof(double));
            double *layerLoss = calloc(_layerSize, sizeof(double));
            // sum the surrounding words
            for (int j = start; j < 2 * _windowSize + 1 - start; j ++) if(j != _windowSize){
                int cindex = i - _windowSize + j;
                if (cindex < 0 || cindex >= senPos) {
                    continue;
                }
                int cword = sentence[cindex];
                vDSP_vaddD(layer, 1, (_neur1 + _layerSize * cword), 1, layer, 1, _layerSize);
            }
            
            // update neur2
            int label, seed;
            for (int l = 0; l <= _negativeSeed; l++) {
                if (l == 0) {
                    label = 1;
                    seed = word;
                }
                else
                {
                    seed = _nsTable[rand() % _loader.totalWords];
                    if (seed == word) {
                        continue;
                    }
                    label = 0;
                }
                double multX = 0;
                for (int p = 0; p < _layerSize; p++) {
                    multX += layer[p] * _neur2[seed*_layerSize + p];
                }
                double q = [self sigmoid:multX];
                double g = _descentRate * (label - q);
                for (int p = 0; p < _layerSize; p++) {
                    layerLoss[p] += g * _neur2[seed*_layerSize + p];
                }
                
                for (int p = 0; p < _layerSize; p++) {
                    _neur2[seed*_layerSize + p] += g * layer[p];
                }
            }
            
            // update neur1
            for (int j = start; j < 2 * _windowSize + 1 - start; j ++) if(j != _windowSize){
                int cindex = i - _windowSize + j;
                if (cindex < 0 || cindex >= senPos) {
                    continue;
                }
                int cword = sentence[cindex];
                vDSP_vaddD(layerLoss, 1, (_neur1 + _layerSize * cword), 1, (_neur1 + _layerSize * cword), 1, _layerSize);
            }
            
            free(layer);
            free(layerLoss);
        }
        
        senPos = 0;
        fseek(file, _loader.fileSize / _threadNum * threadId, SEEK_SET);
        iter--;
    }
}

- (double)sigmoid:(double)x
{
    if (x > 20) {
        return 1;
    }
    if (x < -20) {
        return 0;
    }
    return exp(x) / (1 + exp(x));
}

- (void)createNSTable
{
    _nsTable = calloc(_loader.totalWords, sizeof(int));
    int seg = 0;
    for (int i = 0; i < _loader.dicSize; i++) {
        for (int j = 0; j < _loader.dictionary[i].frequency; j++) {
            _nsTable[seg] = i;
//            printf("%d ",_nsTable[seg]);
            seg++;
        }
    }
}

- (void)testModel
{
    printf("------start test------\n");
    char *input = calloc(MAX_WORDLENGTH, 1);
    while (1) {
        scanf("%s",input);
        if (!strcmp(input, "EXIT")) {
            break;
        }
        int index = [_loader findWordInDict:input];
        if (index == -1) {
            printf("Word Not Found!\n");
            continue;
        }
        else
        {
            printf("Word_Position:%d \n\n",index);
        }
        
        int N = 10; // Get Top 10 closest Word
        double distance[N];
        char *closestWord[N];
        for (int i = 0;  i < N; i++) {
            distance[i] = 0;
            closestWord[i] = "";
        }
        
        double tLen;
        vDSP_svesqD((_neur1 + index * _layerSize), 1, &tLen, _layerSize);
        tLen = sqrt(tLen);
        double *mult = calloc(_layerSize, sizeof(double));
        
        int maxIndex = 0;
        double maxP = 0;
        for (int a = 0; a < _loader.dicSize; a++) if(a != index){
            double pLen;
            vDSP_svesqD((_neur1 + a * _layerSize), 1, &pLen, _layerSize);
            pLen = sqrt(pLen);
            vDSP_vmulD((_neur1 + index * _layerSize), 1, (_neur1 + a * _layerSize), 1, mult, 1, _layerSize);
            double sumMult;
            vDSP_sveD(mult, 1, &sumMult, _layerSize);
            double cosLen = sumMult / tLen / pLen;
            double sim = 0.5 + 0.5 * cosLen;
//            printf("%f\n ",sim);
//            int index = a;
            for (int  i = 0; i < N; i++) {
//                double tempD = distance[i];
//                char *tempW = closestWord[i];
//                if (distance[i] < sim) {
//                    distance[i] = sim;
//                    closestWord[i] = _loader.dictionary[index].word;
//                    index = [_loader findWordInDict:tempW];
//                    sim = tempD;
//                }
                if (sim > distance[i]) {
                    for (int b = N - 1; b > i; b--) {
                        distance[b] = distance[b-1];
                        closestWord[b] = closestWord[b-1];
                    }
                    distance[i] = sim;
                    closestWord[i] = _loader.dictionary[a].word;
                    break;
                }
            }
            
            
            if (sim > maxP) {
                maxP = sim;
                maxIndex = a;
            }
        }
        for (int i = 0; i < N; i++) {
            printf("%s %f\n",closestWord[i],distance[i]);
        }
//        printf("%s %f\n", _loader.dictionary[maxIndex].word, maxP);
        free(mult);
    }
    
}

@end
