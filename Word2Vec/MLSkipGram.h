//
//  MLSkipGram.h
//  Word2Vec
//
//  Created by Jiao Liu on 11/4/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "MLFileLoader.h"

#define MAX_SENTENCE 1000

@interface MLSkipGram : NSObject
{
@private
    double *_neur1;
    double *_neur2;
    MLFileLoader *_loader;
    long long _trainedNum;
    long long _descentGap; //  start from 1e4
    double _startDescentRate;
    int *_nsTable;
}

- (id)initWithLayerSize:(int)size iterNum:(int)num descentRate:(double)rate;
- (void)startTrain:(NSString *)filePath;
- (void)saveToFile:(NSString *)filePath;
- (void)testModel;

@property (nonatomic, assign)int iterNum; // default is 5
@property (nonatomic, assign)int layerSize; // default is 50
@property (nonatomic, assign)long long threadNum; // default is 20
@property (nonatomic, assign)double descentRate; // default is 0.025
@property (nonatomic, assign)int windowSize; // default is 10
@property (nonatomic, assign)bool isHS; // default is Yes
@property (nonatomic, assign)int negativeSeed; //default is 10

@end
