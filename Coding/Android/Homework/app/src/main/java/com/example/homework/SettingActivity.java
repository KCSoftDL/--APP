package com.example.homework;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

public class SettingActivity extends AppCompatActivity {
    private Button setting;
    private Button setting_person;
    private Button setting_support_us;
    private Button setting_feedback;
    private Button setting_contact_us;
    private Button setting_switch_account;
    private static final String ARG_SHOW_TEXT = "text";
    @Override
    public void onBackPressed(){
        Intent intent = new Intent();
        setResult(RESULT_OK,intent);
        finish();
    }
    public static SettingActivity newInstance(String param1) {
        SettingActivity fragment = new SettingActivity();
        Bundle args = new Bundle();
        args.putString(ARG_SHOW_TEXT, param1);
        fragment.setArguments(args);
        return fragment;
    }

    private void setArguments(Bundle args) {
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_setting);
        Toolbar toolbar=(Toolbar)findViewById(R.id.toolbar_setting);
        setSupportActionBar(toolbar);
        toolbar.setTitle("设置"); //标题
        //  toolbar.setSubtitle("");  副标题
        setSupportActionBar(toolbar);
        setting_person = (Button) findViewById(R.id.setting_person);
        setting_person.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Toast.makeText(SettingActivity.this, "该功能尚在开发中", Toast.LENGTH_SHORT).show();
            }
        });
        setting_support_us = (Button) findViewById(R.id.setting_support_us);
        setting_support_us.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Toast.makeText(SettingActivity.this, "该功能尚在开发中", Toast.LENGTH_SHORT).show();
            }
        });
        setting_feedback = (Button) findViewById(R.id.setting_feedback);
        setting_feedback.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Toast.makeText(SettingActivity.this, "该功能尚在开发中", Toast.LENGTH_SHORT).show();
            }
        });
        setting_contact_us = (Button) findViewById(R.id.setting_contact_us);
        setting_contact_us.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Toast.makeText(SettingActivity.this, "该功能尚在开发中", Toast.LENGTH_SHORT).show();
            }
        });
        setting_switch_account = (Button) findViewById(R.id.setting_switch_account);
        setting_switch_account.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Toast.makeText(SettingActivity.this, "该功能尚在开发中", Toast.LENGTH_SHORT).show();
            }
        });
    }
}
