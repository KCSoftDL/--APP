package com.example.homework;

import android.content.Intent;
import android.os.Build;
import android.support.annotation.RequiresApi;
import android.support.design.widget.CollapsingToolbarLayout;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.MenuItem;
import android.widget.ImageView;
import android.widget.TextView;
import android.support.v7.widget.Toolbar;

import com.bumptech.glide.Glide;


public class FoodActivity extends AppCompatActivity {
    public static final String FOOD_NAME="food_name";
    public static final String FOOD_IMAGE_ID="food_image_id";
    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_food);
        Intent intent = getIntent();
        String foodName = intent.getStringExtra(FOOD_NAME);
        int foodImageId = intent.getIntExtra(FOOD_IMAGE_ID, 0);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        CollapsingToolbarLayout collapsingToolbar = (CollapsingToolbarLayout) findViewById(R.id.collapsing_toolbar);
        ImageView foodImageView = (ImageView)findViewById(R.id.food_image_view);
        TextView foodContentText=(TextView)findViewById(R.id.food_content_text);
        setSupportActionBar(toolbar);
        ActionBar actionBar = getSupportActionBar();
        if (actionBar != null) {
            actionBar.setDisplayHomeAsUpEnabled(true);
        }
        collapsingToolbar.setTitle(foodName);
        Glide.with(this).load(foodImageId).into(foodImageView);
        String foodContent = generateFoodContent(foodName);
        foodContentText.setText(foodContent);
    }

    private String generateFoodContent(String foodName) {
        StringBuilder foodContent=new StringBuilder();
            foodContent.append("萝卜做的健康养生食谱\n" +
                    "\n" +
                    "白萝卜+海带：\n" +
                    "\n" +
                    "海带和紫菜含碘丰富。白萝卜和海带一起煮汤有化痰消肿的功效，对预防甲状腺肿大有一定功效。\n" +
                    "\n" +
                    "白萝卜+葱：\n" +
                    "\n" +
                    "冬季，是风寒感冒多发的时节。感染风寒后，人会怕冷、怕风、出汗少、鼻塞流涕、咳嗽有痰，此时应吃些让人发热的食物。把葱段、姜片和白萝卜片一起煮汤，有散寒、止咳的功效，这道汤还可预防感冒。\n" +
                    "\n" +
                    "白萝卜+海带：\n" +
                    "\n" +
                    "海带和紫菜含碘丰富。白萝卜和海带一起煮汤有化痰消肿的功效，对预防甲状腺肿大有一定功效。");
        return foodContent.toString();
    }
    @Override
    public boolean onOptionsItemSelected(MenuItem item){
        switch(item.getItemId()){
            case android.R.id.home:
                finish();
                return true;
        }
        return super.onOptionsItemSelected(item);
    }
}
