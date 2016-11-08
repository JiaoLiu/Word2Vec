//
//  main.m
//  Word2Vec
//
//  Created by Jiao Liu on 11/2/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "MLCBow.h"
#import "MLSkipGram.h"

int main(int argc, const char * argv[]) {
    @autoreleasepool {
//        MLCBow *cbow = [[MLCBow alloc] init];
//        cbow.isHS = NO;
//        [cbow startTrain:@"/Users/Jiao/Desktop/Word2Vec/data.txt"];
////        [cbow saveToFile:@"/Users/Jiao/Desktop/Word2Vec/cbow.txt"];
//        [cbow testModel];
        MLSkipGram *skipGram = [[MLSkipGram alloc] init];
//        skipGram.isHS = NO;
        [skipGram startTrain:@"/Users/Jiao/Desktop/Word2Vec/questions-words.txt"];
        [skipGram testModel];
    }
    return 0;
}
