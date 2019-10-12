package com.example.homework;

import android.os.Build;
import android.os.Bundle;
import android.support.annotation.RequiresApi;
import android.support.v4.app.Fragment;
import android.support.v4.widget.DrawerLayout;
import android.support.v4.widget.SwipeRefreshLayout;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.GridLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.support.v7.widget.Toolbar;
import android.util.SparseArray;
import android.view.MenuItem;
import android.widget.Button;
import android.widget.RadioGroup;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class RecommendActivity extends AppCompatActivity {
    private DrawerLayout mDrawerLayout;   //滑动菜单
    private BiaoQingBao[]biaoQingBaos={new BiaoQingBao("白领如何健康养生",R.drawable.a),new BiaoQingBao("小暑养生食谱 夏季养生",R.drawable.b),new BiaoQingBao("冬季养生精选五道养生食谱",R.drawable.c),new BiaoQingBao("煲汤食谱大全",R.drawable.d),new BiaoQingBao("萝卜做的健康养生食谱",R.drawable.e)};
    private List<BiaoQingBao> biaoQingBaoList=new ArrayList<>();
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
    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    protected void onCreate(Bundle savedInstanceState){
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recommend);
        initBiaoQingBaos();

        Toolbar toolbar=(Toolbar)findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
//        mDrawerLayout=(DrawerLayout)findViewById(R.id.drawer_layout);
//        ActionBar actionBar=getSupportActionBar();
//        if(actionBar!=null){
//            actionBar.setDisplayHomeAsUpEnabled(true);
//            actionBar.setHomeAsUpIndicator(R.drawable.slip_menu);
//        }
//        setting = navView.getHeaderView(0).findViewById(R.id.portrait);
//        setting.setOnClickListener(new View.OnClickListener(){
//            @Override
//            public void onClick(View view){
//                Intent intent=new Intent(RecommendActivity.this,SettingActivity.class);
//                startActivityForResult(intent,3);
//            }
//        });
//
//        nav_take_photo=navView.getMenu().getItem(0);
//        nav_take_photo.setOnMenuItemClickListener(new MenuItem.OnMenuItemClickListener() {
//            @Override
//            public boolean onMenuItemClick(MenuItem item) {
//                Toast.makeText(RecommendActivity.this,"该功能尚在开发中",Toast.LENGTH_SHORT).show();
//                return true;
//            }
//        });
//        nav_share=navView.getMenu().getItem(1);
//        nav_share.setOnMenuItemClickListener(new MenuItem.OnMenuItemClickListener() {
//            @Override
//            public boolean onMenuItemClick(MenuItem item) {
//                Toast.makeText(RecommendActivity.this,"该功能尚在开发中",Toast.LENGTH_SHORT).show();
//                return true;
//            }
//        });
//        nav_health_condition=navView.getMenu().getItem(2);
//        nav_health_condition.setOnMenuItemClickListener(new MenuItem.OnMenuItemClickListener() {
//            @Override
//            public boolean onMenuItemClick(MenuItem item) {
//                Toast.makeText(RecommendActivity.this,"该功能尚在开发中",Toast.LENGTH_SHORT).show();
//                return true;
//            }
//        });
//        nav_nutrition_information=navView.getMenu().getItem(3);
//        nav_nutrition_information.setOnMenuItemClickListener(new MenuItem.OnMenuItemClickListener() {
//            @Override
//            public boolean onMenuItemClick(MenuItem item) {
//                Toast.makeText(RecommendActivity.this,"该功能尚在开发中",Toast.LENGTH_SHORT).show();
//                return true;
//            }
//        });
//        nav_recomment_recipe=navView.getMenu().getItem(4);
//        nav_recomment_recipe.setOnMenuItemClickListener(new MenuItem.OnMenuItemClickListener() {
//            @Override
//            public boolean onMenuItemClick(MenuItem item) {
//                Toast.makeText(RecommendActivity.this,"该功能尚在开发中",Toast.LENGTH_SHORT).show();
//                return true;
//            }
//        });
//        nav_contact_us=navView.getMenu().getItem(5);
//        nav_contact_us.setOnMenuItemClickListener(new MenuItem.OnMenuItemClickListener() {
//            @Override
//            public boolean onMenuItemClick(MenuItem item) {
//                Toast.makeText(RecommendActivity.this,"该功能尚在开发中",Toast.LENGTH_SHORT).show();
//                return true;
//            }
//        });

//        navView.setCheckedItem(R.id.nav_take_photo);
//        navView.setNavigationItemSelectedListener(new NavigationView.OnNavigationItemSelectedListener() {
//            @Override
//            public boolean onNavigationItemSelected(@NonNull MenuItem menuItem) {
//                return true;
//            }
//        });
        swipeRefreshLayout=(SwipeRefreshLayout)findViewById(R.id.swipe_refresh);
        swipeRefreshLayout.setColorSchemeResources(R.color.colorPrimary);
        swipeRefreshLayout.setOnRefreshListener(new SwipeRefreshLayout.OnRefreshListener() {
            @Override
            public void onRefresh() {
                refreshBiaoQingBaos();
            }
        });
        RecyclerView recyclerView=(RecyclerView)findViewById(R.id.recycler_view);
        GridLayoutManager layoutManager=new GridLayoutManager(this,2);
        recyclerView.setLayoutManager(layoutManager);
        adapter=new BiaoQingBaoAdapter(biaoQingBaoList);
        recyclerView.setAdapter(adapter);
    }
    private void refreshBiaoQingBaos(){
        new Thread(new Runnable() {
            @Override
            public void run() {
                try{
                    Thread.sleep(2000);
                }catch (InterruptedException e){
                    e.printStackTrace();
                }
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        initBiaoQingBaos();
                        adapter.notifyDataSetChanged();
                        swipeRefreshLayout.setRefreshing(false);
                    }
                });
            }
        }).start();
    }
    private void initBiaoQingBaos(){
        biaoQingBaoList.clear();
        for(int i=0;i<2;i++){
            Random random=new Random();
            int index=random.nextInt(biaoQingBaos.length);
            biaoQingBaoList.add(biaoQingBaos[index]);
        }
    }
//    @Override
//    public boolean onOptionsItemSelected(MenuItem item) {
//        switch (item.getItemId()) {
//            case android.R.id.home:
//                mDrawerLayout.openDrawer(GravityCompat.START);
//                break;
//            default:
//        }
//        return true;
//    }
}
