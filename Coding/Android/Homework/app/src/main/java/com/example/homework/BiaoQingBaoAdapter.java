package com.example.homework;

import android.content.Context;
import android.content.Intent;
import android.support.annotation.NonNull;
import android.support.v7.widget.CardView;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import com.bumptech.glide.Glide;

import java.util.List;

public class BiaoQingBaoAdapter extends RecyclerView.Adapter<BiaoQingBaoAdapter.ViewHolder> {
    private Context mContext;
    private List<BiaoQingBao> mBiaoQingBaoList;
    static class ViewHolder extends RecyclerView.ViewHolder{
        CardView cardView;
        ImageView biaoqingbaoImage;
        TextView bingqingbaoName;
        public ViewHolder(View view){
            super(view);
            cardView=(CardView)view;
            biaoqingbaoImage=(ImageView)view.findViewById(R.id.biaoqingbao_image);
            bingqingbaoName=(TextView)view.findViewById(R.id.biaoqingbao_name);
        }
    }
    public BiaoQingBaoAdapter(List<BiaoQingBao>biaoQingBaoList){
        mBiaoQingBaoList=biaoQingBaoList;
    }
    @Override
    public ViewHolder onCreateViewHolder(ViewGroup parent,int viewType){
        if(mContext==null){
            mContext=parent.getContext();
        }
        View view = LayoutInflater.from(mContext).inflate(R.layout.biaoqingbao_item,parent,false);
        final ViewHolder holder=new ViewHolder(view);
        holder.cardView.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                int position=holder.getAdapterPosition();
                BiaoQingBao biaoQingBao=mBiaoQingBaoList.get(position);
                Intent intent=new Intent(mContext,FoodActivity.class);
                intent.putExtra(FoodActivity.FOOD_NAME,biaoQingBao.getName());
                intent.putExtra(FoodActivity.FOOD_IMAGE_ID,biaoQingBao.getImageId());
                mContext.startActivity(intent);
            }
        });
        return holder;
    }

    @Override
    public void onBindViewHolder(ViewHolder holder,int position) {
        BiaoQingBao biaoQingBao=mBiaoQingBaoList.get(position);
        holder.bingqingbaoName.setText(biaoQingBao.getName());
        Glide.with(mContext).load(biaoQingBao.getImageId()).into(holder.biaoqingbaoImage);

    }

    @Override
    public int getItemCount() {
       return mBiaoQingBaoList.size();
    }
}
