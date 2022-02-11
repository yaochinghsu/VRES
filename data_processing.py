#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch

def get_data(df):
    df.dropna(inplace = True)
    df.reset_index(drop=True, inplace = True)
    ball = df[df["team_id"]==-1]
    ball = ball[ball["player_id"]== -1]
    skip=[]  ## skip record that do not have 10 members and a ball
    for i in range(1, ball.shape[0]):
        if (ball.index[i]-ball.index[i-1]!= 11):
            skip_index = list(range(ball.index[i-1], ball.index[i]))
            skip += skip_index
    df.drop(skip, inplace = True)
    df.reset_index(drop = True, inplace = True)
    one_go = df[df["team_id"]==-1]
    one_go =one_go[one_go["player_id"]==-1]
    one_go = one_go[one_go["shot_clock"] == 24.0]
    skip=[]  ## skip records that repeats in Sec 24
    for i in range(1, one_go.shape[0]):
        if (one_go.index[i]-one_go.index[i-1]==11):
            skip_index = list(range(one_go.index[i-1], one_go.index[i]))
            skip+=skip_index
    df.drop(skip, inplace = True)
    df.reset_index(drop = True, inplace= True)
    df = df.drop(["team_id", "player_id", "radius", "game_clock", "quarter", "game_id", "event_id"], axis = 1)
    games =[] ## each game suppose to count from 24.0 sec.  However, some game may start somehow 23.98 or 23.95 etc. Clean them
    for i in range(len(df)-1):
        if (i%11 ==0 and df.at[i, "shot_clock"]>23.8):
            games +=[i]
            
    games_flag = [0]
    for i in range(1,len(games)):
        if (games[i]-games[i-1]!=11):
            games_flag +=[games[i]]
    df = df.drop(["shot_clock"], axis = 1)
   
    return df.to_numpy(), np.array(games_flag)



def get_sources_targets(game_data, flag_index):
    
    start_index = flag_index[:-1].copy()
    end_index = flag_index[1:].copy()
    for i in range(len(flag_index)-1): 
        if ((end_index[i]-start_index[i])%2 !=0):
            end_index[i] -=11
    middle_index =((start_index + end_index)/2).astype(int)
    
    sources=[]
    targets=[]
        
    for i in range(len(flag_index)-1):
        source = torch.tensor(game_data[start_index[i]: middle_index[i]]).view(-1, 11, 2).float()
        target = torch.tensor(game_data[middle_index[i]:end_index[i]]).view(-1, 11, 2).float()      
        sources.append(source)
        targets.append(target)
        
    return sources, targets

def get_sources_targets_short(game_data, flag_index, data_size):
    sources=[]
    targets=[]
    data_size = data_size *11
    
    for i in range(len(flag_index)-1):
        
        if (flag_index[i]+ 2*data_size+11) >= flag_index[i+1]:
              continue
        source = torch.tensor(game_data[flag_index[i]: flag_index[i]+ data_size]).view(-1, 11, 2).float()
        target = torch.tensor(game_data[flag_index[i]+ data_size:flag_index[i]+ 2*data_size]).view(-1, 11, 2).float()
        
        sources.append(source)
        targets.append(target)
        
    return sources, targets

def get_speed_short(game_data, flag_index, data_size):
    
    totals =[]
    s_speed =torch.zeros(data_size, 11, 2)
    t_speed =torch.zeros(data_size, 11, 2)
       
    s_speeds = []
    t_speeds = []
    for i in range(len(flag_index)-1):      
        if (flag_index[i]+ 2*11*data_size +11) >= flag_index[i+1]:
              continue
        total = torch.tensor(game_data[flag_index[i]: flag_index[i]+ 2*11*data_size +11]).view(-1, 11, 2).float()
        
        
        for j in range(len(total)-1):
            if j < data_size:
                s_speed[j] = total[j+1][..., :]-total[j][..., :]  
            t_speed[j-data_size] = total[j+1][..., :]-total[j][..., :]  
         
        s_speeds.append(s_speed)
        t_speeds.append(t_speed)
    
    return s_speeds, t_speeds

def get_source_target(sources, targets, s_ordinal=0, s_count=1, t_ordinal = 0, t_count=1):
    
    s = []
    t = []
    for source in sources:
        temp = source[:, s_ordinal: s_ordinal+s_count, :].float()
        s.append(temp)
            
    for target in targets:
        temp = target[:, t_ordinal: t_ordinal+t_count, :].float()
        t.append(temp)
      
    return s, t

def get_with_speed(sources, s_speeds, s_ordinal, s_count,targets, t_speeds, t_ordinal, t_count):
    
    s = []
    t = []
    
    for i in range(len(sources)):
        temp = sources[i][:, s_ordinal: s_ordinal+s_count, :].float()
        temp_s = s_speeds[i][:, s_ordinal: s_ordinal+s_count, :].float()
        s.append(torch.cat((temp, temp_s), 2))
            
    for i in range(len(targets)):
        temp = targets[i][:, t_ordinal: t_ordinal+t_count, :].float()
        temp_s = t_speeds[i][:, t_ordinal: t_ordinal+t_count, :].float()
        t.append(torch.cat((temp, temp_s), 2))
      
    return s, t

def normalize(source, target):
    for s in source:
        s[:, :, 0] = s[:, :, 0]/94.
        s[:, :, 1] = s[:, :, 1]/50.
    for t in target:
        t[:, :, 0] = t[:, :, 0]/94.
        t[:, :, 1] = t[:, :, 1]/50.
      
    return source, target


def reduce_size(sources, targets, size = 100):
    reduce_sources =[]
    for s in sources:
        if len(s)>=size:
            reduce_sources.append(s[-size:])
    reduce_targets = []
    for t in targets:
        if len(t)>=size:
            reduce_targets.append(t[:size])
    sources =reduce_sources
    targets =reduce_targets

    
    return sources, targets


def get_speed(sources, targets):
    s_speed = []
    t_speed = []
    last_ref =torch.zeros(sources[0][0].shape)
    
    for target in targets:
        temp_arr = torch.zeros(target.shape).repeat(1, 1, 2)
        last_ref = target[0]
        for i in range(len(target)-1):
            temp_speed = target[i+1][..., :]-target[i][..., :]
            temp_arr[i] = torch.cat((target[i], temp_speed), -1)
        t_speed.append(temp_arr)
        
    for source in sources:
        temp_arr = torch.zeros(source.shape).repeat(1, 1, 2)        
        for i in range(len(source)-1):
            temp_speed = source[i+1][..., :]-source[i][..., :]           
            temp_arr[i] = torch.cat((source[i], temp_speed), -1)
        temp_arr[-1] = torch.cat((source[-1], last_ref -source[-1][..., :]), -1)
        s_speed.append(temp_arr)
        
       
    return s_speed, t_speed
        