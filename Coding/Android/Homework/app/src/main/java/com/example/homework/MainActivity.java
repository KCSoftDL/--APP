package com.example.homework;

import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.annotation.RequiresApi;
import android.support.design.widget.NavigationView;
import android.support.v4.app.Fragment;
import android.support.v4.view.GravityCompat;
import android.support.v4.widget.DrawerLayout;
import android.support.v4.widget.SwipeRefreshLayout;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.util.SparseArray;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.RadioGroup;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements View.OnClickListener{
    private DrawerLayout mDrawerLayout;
    private BiaoQingBao[]biaoQingBaos={new BiaoQingBao("白领如何健康养生",R.drawable.a),new BiaoQingBao("小暑养生食谱 夏季养生",R.drawable.b),new BiaoQingBao("冬季养生精选五道养生食谱",R.drawable.c),new BiaoQingBao("煲汤食谱大全",R.drawable.d),new BiaoQingBao("萝卜做的健康养生食谱",R.drawable.e)};
    private List<BiaoQingBao>biaoQingBaoList=new ArrayList<>();
    private BiaoQingBaoAdapter adapter;
    private SwipeRefreshLayout swipeRefreshLayout;
    private Button setting;
    private MenuItem nav_take_photo;
    private MenuItem nav_share;
    private MenuItem nav_health_condition;
    private MenuItem nav_nutrition_information;
    private MenuItem nav_recomment_recipe;
    private MenuItem nav_contact_us;
    private RadioGroup mTabRadioGroup;
    private SparseArray<Fragment> mFragmentSparseArray;
    private Button test;
    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @Override
    protected void onActivityResult(int requestCode,int resultCode,Intent data){
        switch(requestCode){
            case 1:
                if(requestCode==RESULT_OK){
                    String returnedData=data.getStringExtra("data_return ");
                    //Log.d
                }
                break;
            default:
        }
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        //initView();
        NavigationView navView=(NavigationView)findViewById(R.id.nav_view);
        mDrawerLayout=(DrawerLayout)findViewById(R.id.drawer_layout);
        ActionBar actionBar=getSupportActionBar();
        if(actionBar!=null){
            actionBar.setDisplayHomeAsUpEnabled(true);
            actionBar.setDisplayShowHomeEnabled(true);
            actionBar.setHomeAsUpIndicator(R.drawable.slip_menu);
        }
        test=(Button)findViewById(R.id.test) ;
        test.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(MainActivity.this,RecommendActivity.class));
            }
        });
        //设置大按钮
        findViewById(R.id.sign_iv).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(MainActivity.this, CameraActivity.class));
            }
        });
        navView.setCheckedItem(R.id.nav_take_photo);
        navView.setNavigationItemSelectedListener(new NavigationView.OnNavigationItemSelectedListener() {
            @Override
            public boolean onNavigationItemSelected(@NonNull MenuItem menuItem) {
                return true;
            }
        });


        //setting=(Button)navView.findViewById(R.id.portrait);

        setting = navView.getHeaderView(0).findViewById(R.id.portrait);
        setting.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view){
                Intent intent=new Intent(MainActivity.this,SettingActivity.class);
                startActivityForResult(intent,3);
            }
        });

       nav_take_photo=navView.getMenu().getItem(0);
        nav_take_photo.setOnMenuItemClickListener(new MenuItem.OnMenuItemClickListener() {
            @Override
            public boolean onMenuItemClick(MenuItem item) {
                Toast.makeText(MainActivity.this,"该功能尚在开发中",Toast.LENGTH_SHORT).show();
                return true;
            }
        });
        nav_share=navView.getMenu().getItem(1);
        nav_share.setOnMenuItemClickListener(new MenuItem.OnMenuItemClickListener() {
            @Override
            public boolean onMenuItemClick(MenuItem item) {
                Toast.makeText(MainActivity.this,"该功能尚在开发中",Toast.LENGTH_SHORT).show();
                return true;
            }
        });
        nav_health_condition=navView.getMenu().getItem(2);
        nav_health_condition.setOnMenuItemClickListener(new MenuItem.OnMenuItemClickListener() {
            @Override
            public boolean onMenuItemClick(MenuItem item) {
                Toast.makeText(MainActivity.this,"该功能尚在开发中",Toast.LENGTH_SHORT).show();
                return true;
            }
        });
        nav_nutrition_information=navView.getMenu().getItem(3);
        nav_nutrition_information.setOnMenuItemClickListener(new MenuItem.OnMenuItemClickListener() {
           @Override
           public boolean onMenuItemClick(MenuItem item) {
               Toast.makeText(MainActivity.this,"该功能尚在开发中",Toast.LENGTH_SHORT).show();
               return true;
           }
       });
        nav_recomment_recipe=navView.getMenu().getItem(4);
        nav_recomment_recipe.setOnMenuItemClickListener(new MenuItem.OnMenuItemClickListener() {
            @Override
            public boolean onMenuItemClick(MenuItem item) {
                Toast.makeText(MainActivity.this,"该功能尚在开发中",Toast.LENGTH_SHORT).show();
                return true;
            }
        });
       nav_contact_us=navView.getMenu().getItem(5);
        nav_contact_us.setOnMenuItemClickListener(new MenuItem.OnMenuItemClickListener() {
            @Override
            public boolean onMenuItemClick(MenuItem item) {
                Toast.makeText(MainActivity.this,"该功能尚在开发中",Toast.LENGTH_SHORT).show();
                return true;
            }
        });

        navView.setCheckedItem(R.id.nav_take_photo);
        navView.setNavigationItemSelectedListener(new NavigationView.OnNavigationItemSelectedListener() {
            @Override
            public boolean onNavigationItemSelected(@NonNull MenuItem menuItem) {
               return true;
            }
        });

//        FloatingActionButton fab=(FloatingActionButton)findViewById(R.id.fab);
//        fab.setOnClickListener(new View.OnClickListener(){
//            @Override
//            public void onClick(View view){
//                Snackbar.make(view,"拍照识别",Snackbar.LENGTH_SHORT).setAction("我点错了",new View.OnClickListener(){
//                    @Override
//                    public void onClick(View v){
//                        Toast.makeText(MainActivity.this,"哼唧，乱点什么嘛",Toast.LENGTH_SHORT).show();
//                    }
//                }).show();
//            }
//        });

}

   private void initView() {
        mTabRadioGroup = findViewById(R.id.tabs_rg);
        mFragmentSparseArray = new SparseArray<>();
        mFragmentSparseArray.append(R.id.today_tab, HealthConditionActivity.newInstance("健康状况"));
        mFragmentSparseArray.append(R.id.record_tab, RecommendRecipeActivity.newInstance("推荐食谱"));
        mFragmentSparseArray.append(R.id.contact_tab, ShareActivity.newInstance("分享圈"));
        mFragmentSparseArray.append(R.id.settings_tab, BlankFragment.newInstance("设置"));
        mTabRadioGroup.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                // 具体的fragment切换逻辑可以根据应用调整，例如使用show()/hide()
                getSupportFragmentManager().beginTransaction().replace(R.id.fragment_container,
                        mFragmentSparseArray.get(checkedId)).commit();
            }
        });
        // 默认显示第一个
        getSupportFragmentManager().beginTransaction().add(R.id.fragment_container,
                mFragmentSparseArray.get(R.id.today_tab)).commit();

    }


    @Override
    public void onClick(View v) {
        switch (v.getId()) {

            default:
                break;
        }
    }
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                mDrawerLayout.openDrawer(GravityCompat.START);
                break;
            default:
        }
        return true;
    }
}